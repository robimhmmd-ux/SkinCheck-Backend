"""
AI Assistant Service for SkinCheck (Asisten Pribadi Luvenna / Ven)
Architecture: SafetyLayer → ContextBuilder → RAG → LLM (Google Gemini) → Response

PANDUAN UTAMA CHATBOT LUVENNA (TITAH PADUKA RAJA):
1. PENEMPATAN WIDGET: Chatbot hanya ada di menu dashboard (/dashboard, /customer/dashboard, /admin/dashboard).
2. BATASAN DATABASE: Pengguna hanya berhak melihat data pribadi miliknya. Dilarang melihat data pengguna lain/sistem.
3. GAYA BAHASA NATURAL & RAMAH: Bahasa mengalir alami, luwes, santun, hangat, dan bersahabat layaknya asisten pribadi manusiawi terpercaya, bukan robot kaku ("berdasarkan sistem", menyampaikan data mentah).
4. SAPAAN RAMAH (GREETING): Jika disapa (hi, halo, dsb.), balas ramah dengan menyebut nama pengguna ("Halo [Nama User]! Ada yang bisa Luvenna bantu?").
5. MENJAWAB IDENTITAS DIRI: Jika ditanya "siapa aku/siapa saya", olah kalimat menjadi percakapan personal yang hangat dan menyenangkan (misal "Kamu adalah [Nama], pengguna setia SkinCheck yang sudah melakukan [X] kali scan wajah..."), bukan membacakan tabel paspor mentah.
6. TIDAK LEBAY: Santun dan profesional, tanpa rentetan emotikon lebay (✨, 🌿, 🤗, 💖, 🥰, 🌸, 🌱, 🛒) atau kata alay.
7. NAMA ASISTEN: Luvenna (Ven).
8. FOKUS KULIT WAJAH: Menjawab seputar kesehatan kulit wajah, kandungan skincare, dan rekomendasi Shopee di database. Di luar batasan, minta maaf dan tolak ramah.
9. SINKRONISASI TAMPILAN DASHBOARD (USER & ADMIN): Rerata kondisi kulit (jerawat, minyak, kekeringan, noda hitam, garis halus) WAJIB dalam format PERSEN (%) sama persis dengan dashboard (dilarang pecahan koma desimal seperti 0.15).
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, List
from sqlalchemy.orm import Session

from app.config import settings
from app.models.user import User
from app.models.skin_scan import SkinScan
from app.models.analysis_result import AnalysisResult
from app.models.product import Product
from app.models.system_setting import SystemSetting
from app.models.chat_message import ChatMessage
from app.models.activity_log import ActivityLog
from app.models.otp_token import OTPToken
from sqlalchemy import case, func, desc

logger = logging.getLogger(__name__)


def normalize_chat_question(text: str) -> str:
    """Normalisasi pertanyaan customer untuk pengelompokan: huruf kecil,
    buang tanda baca, rapatkan spasi. Pertanyaan yang sama (kata sama,
    tanda baca/kapital beda) terhitung satu."""
    t = (text or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def top_customer_questions(db: Session, limit: int = 10):
    """Pertanyaan customer (role=user) yang paling sering diajukan ke
    asisten, dikelompokkan berdasarkan teks yang dinormalisasi.
    Mengembalikan list dict {question, count, last_asked_at, asked_by}
    terurut jumlah terbanyak dulu. Dipakai endpoint tren chat admin dan
    fallback chatbot admin — datanya selalu dari ChatMessage, bukan
    daftar contoh."""
    rows = (
        db.query(ChatMessage.content, User.name, ChatMessage.created_at)
        .join(User, ChatMessage.user_id == User.id)
        .filter(User.role == "customer", ChatMessage.role == "user")
        .order_by(desc(ChatMessage.created_at))
        .limit(500)
        .all()
    )
    grouped = {}
    for content, name, created_at in rows:
        key = normalize_chat_question(content)
        if not key or len(key) < 3:
            continue
        g = grouped.get(key)
        if g is None:
            # Baris sudah terurut terbaru dulu: wording pertama = terbaru.
            g = grouped[key] = {
                "question": (content or "").strip(),
                "count": 0,
                "last_asked_at": created_at,
                "asked_by": [],
            }
        g["count"] += 1
        if name and name not in g["asked_by"]:
            g["asked_by"].append(name)
    out = sorted(
        grouped.values(),
        key=lambda g: (
            -g["count"],
            -(g["last_asked_at"].timestamp() if g["last_asked_at"] else 0),
        ),
    )
    return out[: max(1, limit)]


def find_user_by_name(db: Session, msg_lower: str, role: str | None = None):
    """Cari user yang namanya disebut di pesan (nama lengkap atau kata
    pertama nama). Dipakai fallback chatbot admin untuk 'baca chat <nama>'
    dan 'berapa kali <nama> scan'."""
    q = db.query(User.id, User.name, User.email)
    if role:
        q = q.filter(User.role == role)
    for u in q.all():
        for cand in {u.name.lower(), u.name.lower().split()[0]}:
            if cand and len(cand) >= 3 and re.search(r"\b" + re.escape(cand) + r"\b", msg_lower):
                return u
    return None


# ─────────────────────────────────────────────────────────
# 1. SAFETY & DOMAIN GUARDRAIL LAYER
# ─────────────────────────────────────────────────────────

class SafetyLayer:
    """
    Pre- and post-filter for high-risk medical topics and off-topic domain enforcement.
    Elevated privileges applied when role == 'admin' to allow querying all website database tables.
    """

    HIGH_RISK_KEYWORDS = [
        "luka terbuka", "berdarah", "perdarahan", "infeksi berat", "nanah banyak",
        "pembengkakan bibir", "bengkak parah", "anafilaksis", "sesak nafas",
        "resep dokter", "prescription", "obat keras", "dosis obat",
        "antibiotik minum", "oral steroid", "isotretinoin", "accutane",
        "kanker kulit", "melanoma", "tumor", "biopsi",
        "darurat", "hentikan obat dokter", "berhenti minum obat resep",
    ]

    ESCALATION_RESPONSE = (
        "Kondisi seperti luka terbuka, perdarahan, infeksi berat, bengkak parah, atau obat keras memerlukan penanganan medis langsung. "
        "Luvenna sangat menyarankanmu untuk segera berkonsultasi langsung dengan dokter spesialis dermatologi (Sp.DVE / Sp.KK) agar mendapatkan penanganan yang tepat dan aman.\n\n"
        "SkinCheck AI mendampingi panduan edukasi visual dan rekomendasi perawatan kulit non-medis, "
        "bukan pengganti tindakan dan resep dokter profesional."
    )

    OFF_TOPIC_REJECTION = (
        "Mohon maaf ya, saat ini Luvenna hanya dapat membantu konsultasi seputar kesehatan dan perawatan kulit wajah serta data akun SkinCheck milikmu."
    )

    OFF_TOPIC_PATTERNS = [
        r"\b(python|javascript|golang|php|html|css|sql|coding|program|syntax|function|bug)\b",
        r"\b(jokowi|prabowo|pemilu|pilpres|dpr|partai|politik)\b",
        r"\b(kalkulus|integral|turunan|aljabar|rumus fisika|trigonometri)\b",
        r"\b(resep masakan|cara masak rendang|nasi goreng|bikin kue|resep soto)\b",
        r"\b(bitcoin|crypto|saham|trading|forex|investasi uang)\b",
        r"\b(sepak bola|champions league|premier league|mobile legends|game pass)\b"
    ]

    DATABASE_RESTRICTION_REJECTION = (
        "Demi menjaga privasi dan keamanan bersama, Luvenna hanya bisa mendampingi dan menampilkan data akun serta riwayat kulit milikmu sendiri ya. Data pengguna lain maupun sistem bersifat rahasia."
    )

    UNAUTHORIZED_DB_PATTERNS = [
        r"\b(user|pengguna|akun|member|customer)\s+lain\b",
        r"\b(semua|daftar|seluruh|total)\s+(user|pengguna|akun|member|customer)\b",
        r"\b(database|tabel|db)\s+(project|website|sistem|aplikasi|semua|seluruh)\b",
        r"\b(seluruh|semua)\s+database\b",
        r"\b(tabel\s+admin|data\s+admin|system\s+settings?)\b",
        r"\b(email|password|data)\s+(orang|user|pengguna)\s+lain\b",
        r"\b(siapa\s+saja\s+(yang\s+)?(daftar|terdaftar|punya\s+akun))\b",
        r"\b(bocorkan|tampilkan|lihat)\s+(semua|data)\s+(user|akun)\b",
    ]

    CUSTOMER_ALLOWED_ACCOUNT_KEYWORDS = [
        "profil", "akun", "scan", "pemindaian", "riwayat", "skor", "kesehatan",
        "nama", "email", "tanggal", "aktivitas", "saya", "diri saya", "punya saya",
        "siapa saya", "siapa aku", "siapa dia", "siapa diriku", "kamu tahu siapa", "tentang aku", "tentang saya",
        "berapa kali", "rata-rata", "rata rata", "rerata", "hasil cek", "hasil scan",
        "shopee", "link shopee", "produk shopee", "produk", "rekomendasi", "katalog", "cek", "data",
        "grafik", "dashboard", "persen", "persentase", "indeks", "jerawat", "minyak", "kering", "noda", "flek", "kerutan", "garis halus",
        "hi", "hii", "hiii", "halo", "hai", "helo", "hello", "hey", "hola", "p", "pagi", "siang", "sore", "malam", "assalamualaikum", "ven", "luvenna"
    ]

    SKIN_KEYWORDS = [
        "kulit", "wajah", "muka", "jerawat", "acne", "komedo", "pori", "sebum",
        "minyak", "kering", "kusam", "flek", "noda", "spots", "kerutan", "wrinkle", "garis halus",
        "sunscreen", "tabir surya", "spf", "uv", "skincare", "serum", "toner",
        "moisturizer", "pelembap", "cleanser", "sabun", "cuci muka", "niacinamide",
        "salicylic", "retinol", "ceramide", "hyaluronic", "centella", "cica",
        "shopee", "beli", "produk", "rekomendasi", "barrier", "iritasi", "kemerahan",
        "treatment", "eksfoliasi", "peeling", "bha", "aha", "pha", "vitamin c"
    ]

    @classmethod
    def check_input(cls, message: str, role: str = "customer") -> Optional[str]:
        """Returns escalation or rejection text if high-risk or off-topic, else None.
        Admin has ZERO restrictions (no topic boundaries, full coding and database privileges).
        """
        # Admin has zero topic restrictions (titah Paduka Raja: admin bebas memerintahkan apapun)
        if role == "admin":
            return None

        lower = message.lower().strip()

        # Check greetings first — never reject warm greetings
        greetings = [
            "hi", "hii", "hiii", "halo", "hai", "helo", "hello", "hey", "hola",
            "p", "pagi", "siang", "sore", "malam", "assalamualaikum", "assalam",
            "halo ven", "hai ven", "ven", "luvenna"
        ]
        cleaned_first_word = re.sub(r"[^\w\s]", "", lower).strip().split()
        if cleaned_first_word and cleaned_first_word[0] in greetings:
            return None

        # Check if query is explicitly an authorized personal database, dashboard, or shopee query
        is_allowed_personal_db = any(ak in lower for ak in [
            "siapa saya", "siapa aku", "siapa dia", "siapa diriku", "kamu tahu siapa", "tentang aku", "tentang saya",
            "berapa kali", "hasil cek", "hasil scan", "rata-rata", "rata rata", "rerata",
            "link shopee", "produk shopee", "shopee yang ada di database", "shopee di database",
            "profil saya", "akun saya", "data saya", "riwayat scan", "grafik", "dashboard", "persen",
            "jerawat saya", "minyak saya", "kulit saya", "skor saya"
        ])

        # Pertanyaan definisi soal konsep aplikasi ("apa itu database") bukan
        # akses data — biarkan lolos ke cabang glosarium di fallback, yang
        # hanya berisi pengetahuan umum aplikasi (tanpa data pengguna).
        # _is_definition_question/_match_glossary didefinisikan di bawah di
        # modul ini, tapi dipanggil saat runtime (modul sudah termuat penuh).
        if _is_definition_question(lower) and _match_glossary(lower):
            return None

        # 1. Database scope boundary check for regular customers (prevent accessing others' or system tables)
        if not is_allowed_personal_db:
            for pattern in cls.UNAUTHORIZED_DB_PATTERNS:
                if re.search(pattern, lower):
                    logger.info(f"SafetyLayer triggered on unauthorized db query pattern: '{pattern}'")
                    return cls.DATABASE_RESTRICTION_REJECTION

        # 2. High risk check for regular customers
        for kw in cls.HIGH_RISK_KEYWORDS:
            if kw in lower:
                logger.info(f"SafetyLayer triggered on high-risk keyword: '{kw}'")
                return cls.ESCALATION_RESPONSE

        # 3. Off-topic pattern match for regular customers
        has_skin_term = any(sk in lower for sk in cls.SKIN_KEYWORDS)
        has_account_term = any(ak in lower for ak in cls.CUSTOMER_ALLOWED_ACCOUNT_KEYWORDS)
        if not has_skin_term and not has_account_term:
            for pattern in cls.OFF_TOPIC_PATTERNS:
                if re.search(pattern, lower):
                    logger.info(f"SafetyLayer triggered on off-topic pattern: '{pattern}'")
                    return cls.OFF_TOPIC_REJECTION

        return None

    @classmethod
    def post_filter(cls, response: str, is_pure_greeting: bool = False) -> str:
        """Scrub medical diagnoses, robotic prefixes, unsolicited greetings, and excessive emojis."""
        forbidden = [
            ("Anda menderita", "Anda mungkin mengalami indikasi"),
            ("Anda terkena", "Anda mungkin mengalami indikasi"),
            ("diagnosis medis", "indikasi visual"),
            ("penyakit ini adalah", "kondisi ini mungkin menunjukkan"),
            ("minum obat keras", "konsultasikan dengan dokter spesialis kulit"),
        ]
        result = response
        for bad, good in forbidden:
            result = result.replace(bad, good)

        # Hapus sapaan pembuka dan basa-basi jika pengguna menanyakan sesuatu (bukan sapaan murni)
        if not is_pure_greeting:
            greeting_prefixes = [
                r"^(?:(?:Halo|Hai|Hi|Hello|Hey)(?:\s+(?:\*\*)?[^\n.!?]+?(?:\*\*)?)?[,!.\n]+\s*)",
                r"^(?:(?:Selamat\s+(?:pagi|siang|sore|malam))(?:\s+(?:\*\*)?[^\n.!?]+?(?:\*\*)?)?[,!.\n]+\s*)",
                r"^(?:Senang[^\n.!?]*[.!?\n]+\s*)",
                r"^(?:Terima\s+kasih[^\n.!?]*[.!?\n]+\s*)",
                r"^(?:(?:Tentu|Tentu saja|Baik|Baiklah)[,!\s]+)",
                r"^(?:Berikut\s+(?:ini\s+)?[^\n.!?]*[.!?\n]+\s*)",
                r"^(?:berdasarkan\s+(?:data\s+)?(?:sistem|database|catatan\s+database)[^\n,!?]*[,\n!?]+\s*)",
                r"^(?:terkait\s+(?:dengan\s+)?pertanyaan\s+(?:Anda|kamu|mengenai)[^\n.!?]*[.!?\n]+\s*)",
                r"^(?:sebagai\s+(?:model\s+ai|bot|asisten\s+virtual\s+robot)[^\n,!?]*[,\n!?]+\s*)",
            ]
            prev = None
            while prev != result:
                prev = result
                for pat in greeting_prefixes:
                    result = re.sub(pat, "", result, flags=re.IGNORECASE).strip()

        # Hapus rentetan emotikon lebay
        emojis_to_strip = ["✨", "🌿", "🤗", "💖", "🥰", "🌸", "🌱", "👍", "👏", "🛒", "Hehe,", "hehe,", "Hehe", "hehe"]
        for em in emojis_to_strip:
            result = result.replace(em, "")

        return result.strip()



# ─────────────────────────────────────────────────────────
# 2. RAG KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────

class RAGKnowledgeBase:
    """
    Local knowledge base for trusted skincare education and clinical ingredient guidance.
    """

    def __init__(self):
        self.documents = []
        self._load()

    def _load(self):
        kb_path = Path(__file__).parent.parent / "data" / "skincare_knowledge.json"
        if kb_path.exists():
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                logger.info(f"RAG loaded {len(self.documents)} documents.")
            except Exception as e:
                logger.warning(f"Failed to load RAG knowledge base: {e}")
        else:
            logger.warning(f"Knowledge base not found at {kb_path}")

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """Simple keyword relevance retrieval."""
        query_lower = query.lower()
        scored = []
        for doc in self.documents:
            score = 0
            searchable = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
            for word in query_lower.split():
                if len(word) > 2 and word in searchable:
                    score += 1
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


# ─────────────────────────────────────────────────────────
# 3. CONTEXT BUILDER (User Profile, Scans & Indonesian Products)
# ─────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Builds rich context from user profile, facial condition scan, and the Product Catalog with Shopee links.
    """

    @staticmethod
    def build_user_context(user_id: int, db: Session, scan_id: Optional[int] = None) -> str:
        """
        Builds structured personal context strictly for the logged-in user.
        FAST VERSION (speed fix 2026-11-04): the old version loaded ALL of the
        user's scans and then issued one AnalysisResult query per scan (N+1)
        plus one query per recent scan. This version issues exactly 4 bounded
        queries: profile, aggregate stats, latest analysed scan, recent 5 scans
        with their analyses fetched in a single IN query. Output format is kept
        identical so prompt behaviour does not change.
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return "Data pengguna tidak ditemukan."

        profile_data = user.get_profile_data() if hasattr(user, "get_profile_data") else {}
        skin_concerns = profile_data.get("skin_concerns", [])
        concerns_str = ", ".join(skin_concerns) if skin_concerns else "Belum disetel"
        joined_str = user.created_at.strftime("%d %B %Y") if user.created_at else "-"

        total_scans = db.query(func.count(SkinScan.id)).filter(SkinScan.user_id == user_id).scalar() or 0

        # One aggregate query for averages (same fallbacks as before:
        # dark spots -> hyperpigmentation, wrinkles -> texture, when the
        # primary score is missing/zero).
        dark_expr = case(
            (AnalysisResult.dark_spots_score > 0, AnalysisResult.dark_spots_score),
            else_=AnalysisResult.hyperpigmentation_score,
        )
        wr_expr = case(
            (AnalysisResult.wrinkles_score > 0, AnalysisResult.wrinkles_score),
            else_=AnalysisResult.texture_score,
        )
        agg = (
            db.query(
                func.count(AnalysisResult.id),
                func.avg(AnalysisResult.acne_score),
                func.avg(AnalysisResult.oily_skin_score),
                func.avg(AnalysisResult.dry_skin_score),
                func.avg(dark_expr),
                func.avg(wr_expr),
                func.avg(AnalysisResult.overall_score),
            )
            .join(SkinScan, AnalysisResult.skin_scan_id == SkinScan.id)
            .filter(SkinScan.user_id == user_id, AnalysisResult.overall_score.isnot(None))
            .first()
        )
        analyzed_count = int(agg[0] or 0)

        if analyzed_count > 0:
            avg_acne = round(float(agg[1] or 0), 1)
            avg_oily = round(float(agg[2] or 0), 1)
            avg_dry = round(float(agg[3] or 0), 1)
            avg_spots = round(float(agg[4] or 0), 1)
            avg_wrinkles = round(float(agg[5] or 0), 1)
            avg_overall = round(float(agg[6] or 0), 1)

            acne_status = "Perlu Treat" if avg_acne > 50 else ("Sedang" if avg_acne > 25 else "Terkontrol")
            oily_status = "Berminyak" if avg_oily > 50 else ("Normal" if avg_oily > 25 else "Kering")
            dry_status = "Dehidrasi" if avg_dry > 50 else ("Normal" if avg_dry > 25 else "Lembap")
            spots_status = "Hiperpigmentasi" if avg_spots > 40 else "Minim Noda"
            wrinkles_status = "Perlu Anti-Aging" if avg_wrinkles > 40 else "Elastis"

            if avg_overall >= 75:
                overall_status = "Sangat Sehat"
            elif avg_overall >= 50:
                overall_status = "Kondisi Terawat"
            elif avg_overall >= 35:
                overall_status = "Perlu Hidrasi & Perawatan"
            else:
                overall_status = "Perlu Perhatian Ekstra"

            averages_section = (
                f"2. KARTU & GRAFIK TREN RERATA KONDISI KULIT (SINKRON 100% DENGAN TAMPILAN DASHBOARD USER):\n"
                f"   [PENTING: Nilai rerata 5 parameter kondisi kulit di bawah WAJIB dijawab dalam satuan PERSEN %]:\n"
                f"   • Rerata Jerawat (Acne): {avg_acne}% (Status: {acne_status})\n"
                f"   • Rerata Minyak (Oily): {avg_oily}% (Status: {oily_status})\n"
                f"   • Rerata Kekeringan (Dryness): {avg_dry}% (Status: {dry_status})\n"
                f"   • Rerata Noda Hitam & Flek (Dark Spots): {avg_spots}% (Status: {spots_status})\n"
                f"   • Rerata Garis Halus & Kerutan (Wrinkles): {avg_wrinkles}% (Status: {wrinkles_status})\n"
                f"   • Rerata Indeks Kesehatan Kulit Keseluruhan: {avg_overall}/100 ({overall_status})\n"
                f"   • Total Hasil Scan yang Dianalisis: {analyzed_count} kali (dari {total_scans} total pemindaian)"
            )
        else:
            avg_acne = avg_oily = avg_dry = avg_spots = avg_wrinkles = avg_overall = 0
            averages_section = (
                f"2. KARTU & GRAFIK TREN RERATA KONDISI KULIT (DASHBOARD USER):\n"
                f"   Belum ada riwayat scan wajah yang selesai dianalisis. Total pemindaian: {total_scans} kali."
            )

        # Latest analysed scan: one query (newest scan that has an analysis).
        latest_row = (
            db.query(SkinScan, AnalysisResult)
            .join(AnalysisResult, AnalysisResult.skin_scan_id == SkinScan.id)
            .filter(SkinScan.user_id == user_id, AnalysisResult.overall_score.isnot(None))
            .order_by(desc(SkinScan.created_at))
            .first()
        )
        if latest_row:
            latest_scan, latest_analysis = latest_row
            l_ov = int(latest_analysis.overall_score or 0)
            l_ac = int(latest_analysis.acne_score or 0)
            l_oi = int(latest_analysis.oily_skin_score or 0)
            l_dr = int(latest_analysis.dry_skin_score or 0)
            l_sp = int(latest_analysis.dark_spots_score if latest_analysis.dark_spots_score is not None and latest_analysis.dark_spots_score > 0 else (latest_analysis.hyperpigmentation_score or 0))
            l_wr = int(latest_analysis.wrinkles_score if latest_analysis.wrinkles_score is not None and latest_analysis.wrinkles_score > 0 else (latest_analysis.texture_score or 0))
            l_time = latest_scan.created_at.strftime("%d %B %Y, %H:%M WIB") if latest_scan.created_at else "-"

            latest_scan_section = (
                f"3. ANALISIS KONDISI KULIT TERAKHIR (LATEST SCAN PADA DASHBOARD USER):\n"
                f"   • Waktu Pemeriksaan: {l_time} (Scan #{latest_scan.id})\n"
                f"   • Skor Kesehatan Keseluruhan: {l_ov}/100\n"
                f"   • Jerawat & Inflamasi (Acne): {l_ac}%\n"
                f"   • Tingkat Sebum / Minyak (Oily): {l_oi}%\n"
                f"   • Kekeringan Kulit (Dryness): {l_dr}%\n"
                f"   • Noda Hitam & Flek (Dark Spots): {l_sp}%\n"
                f"   • Garis Halus & Tekstur (Wrinkles): {l_wr}%"
            )
        else:
            latest_scan_section = "3. ANALISIS KONDISI KULIT TERAKHIR:\n   Belum ada scan yang selesai dianalisis."

        # Recent 5 scans + their analyses in a single IN query (no per-scan query).
        recent_scans = (
            db.query(SkinScan)
            .filter(SkinScan.user_id == user_id)
            .order_by(desc(SkinScan.created_at))
            .limit(5)
            .all()
        )
        recent_ids = [s.id for s in recent_scans]
        recent_analyses = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.skin_scan_id.in_(recent_ids))
            .all()
            if recent_ids
            else []
        )
        analysis_by_scan = {}
        for ar in recent_analyses:
            analysis_by_scan.setdefault(ar.skin_scan_id, ar)
        history_lines = []
        for sc in recent_scans:
            ar = analysis_by_scan.get(sc.id)
            sc_date = sc.created_at.strftime("%d %b %Y") if sc.created_at else "-"
            sc_score = f"{int(ar.overall_score)}/100" if ar and ar.overall_score is not None else "-"
            history_lines.append(f"   • Scan #{sc.id} | {sc_date} | Skor: {sc_score} | Status: {sc.status.capitalize()}")
        history_str = "\n".join(history_lines) if history_lines else "   (Belum ada riwayat)"

        masked_email = re.sub(r"(?<=.).(?=.*@)", "*", user.email) if user.email and "@" in user.email else "-"
        return (
            f"DATA PENGGUNA & RINGKASAN DASHBOARD (SINKRON REAL-TIME DENGAN TAMPILAN WEB):\n"
            f"1. PROFIL PENGGUNA:\n"
            f"   • Nama Pengguna: {user.name}\n"
            f"   • Email Terdaftar: {masked_email}\n"
            f"   • Tanggal Bergabung: {joined_str}\n"
            f"   • Status Akun: {user.status}\n"
            f"   • Fokus Perawatan Kulit: {concerns_str}\n\n"
            f"{averages_section}\n\n"
            f"{latest_scan_section}\n\n"
            f"4. RIWAYAT PEMINDAIAN TERAKHIR:\n"
            f"{history_str}\n\n"
            f"PANDUAN MENJAWAB SEPERTI ASISTEN PRIBADI TERPERCAYA (LUVENNA):\n"
            f"- Jawab to the point, padat, dan ramah langsung ke inti pertanyaan (maksimal 1-2 kalimat).\n"
            f"- HANYA JAWAB PARAMETER YANG DITANYAKAN: Jika ditanya jerawat, hanya jawab jerawat (jangan jelaskan minyak/kondisi lain yang tidak ditanyakan).\n"
            f"- JANGAN MENYAPA DI AWAL RESPON (tanpa 'Halo', 'Hai', dsb.).\n"
            f"- Parameter rerata jerawat, minyak, kekeringan, noda hitam, dan garis halus WAJIB dijawab dalam format PERSEN (%) sama persis dengan angka di atas (misal '{avg_acne}%'). Dilarang menggunakan pecahan koma desimal mentah (seperti 0.15)."
        )

    @staticmethod
    def build_admin_database_context(db: Session, query: str = "", admin_user: Optional[dict] = None) -> str:
        """
        Builds a comprehensive live snapshot of the entire SkinCheck database for Admin queries.
        Includes:
        - Exact Admin Dashboard UI metric cards (Customer, Face Scans, Products, Success Rate, AI Sessions)
        - Project-wide table statistics (Users, Scans, Products, Logs, Settings)
        - Disease and skin condition trends across all users (AnalysisResult platform averages)
        - Customer chat inquiries, complaints, and requests trends (extracted from ChatMessage where User.role == 'customer')
        - Feed of recent customer inquiries for direct analysis by Luvenna
        - Identity of the logged-in admin (section 0), so "siapa saya" is never
          answered with a scan ID or a made-up ID.
        """
        try:
            from datetime import date
            today = date.today()

            # 1. Users snapshot
            total_users = db.query(User).count()
            total_customers = db.query(User).filter(User.role == "customer").count()
            total_admins = db.query(User).filter(User.role == "admin").count()
            active_users = db.query(User).filter(User.status == "active").count()
            active_customers = db.query(User).filter(User.role == "customer", User.status == "active").count()
            inactive_users = total_users - active_users
            recent_users = db.query(User).order_by(User.created_at.desc()).limit(8).all()
            user_list_str = "\n".join([
                f"  • ID #{u.id} | {u.name} | {u.email} | Role: {u.role} | Status: {u.status} | Terdaftar: {u.created_at.strftime('%d-%m-%Y %H:%M')}"
                for u in recent_users
            ])

            # 2. Scans snapshot
            total_scans = db.query(SkinScan).count()
            scans_today = db.query(SkinScan).filter(func.date(SkinScan.created_at) == today).count()
            completed_scans = db.query(SkinScan).filter(SkinScan.status == "completed").count()
            failed_scans = db.query(SkinScan).filter(SkinScan.status == "failed").count()
            recent_scans = db.query(SkinScan).order_by(SkinScan.created_at.desc()).limit(5).all()
            scan_list_str = "\n".join([
                f"  • Scan #{s.id} | User ID: #{s.user_id} | Status: {s.status} | Kualitas: {s.image_quality_score or '-'} | Waktu: {s.created_at.strftime('%d-%m-%Y %H:%M')}"
                for s in recent_scans
            ])

            # Analysis success rate
            total_analysis = completed_scans + failed_scans
            success_rate = f"{(completed_scans / total_analysis * 100):.1f}%" if total_analysis > 0 else "0%"

            # Total AI Sessions
            total_ai_sessions = db.query(ChatMessage.user_id).distinct().count()

            # 3. Clinical Skin Condition Averages across all scans
            avg_metrics = db.query(
                func.count(AnalysisResult.id).label("total"),
                func.avg(AnalysisResult.overall_score).label("overall"),
                func.avg(AnalysisResult.acne_score).label("acne"),
                func.avg(AnalysisResult.oily_skin_score).label("oily"),
                func.avg(AnalysisResult.dry_skin_score).label("dry"),
                func.avg(AnalysisResult.dark_spots_score).label("dark_spots"),
                func.avg(AnalysisResult.wrinkles_score).label("wrinkles"),
            ).first()

            avg_summary = (
                f"  - Total Analisis Selesai: {avg_metrics.total or 0} scan\n"
                f"  - Rerata Skor Kesehatan Kulit: {round(avg_metrics.overall or 0, 1)}/100\n"
                f"  - Rerata Tingkat Jerawat (Acne): {round(avg_metrics.acne or 0, 1)}/100 (Masalah paling dominan)\n"
                f"  - Rerata Kulit Kering (Dry): {round(avg_metrics.dry or 0, 1)}/100\n"
                f"  - Rerata Kulit Berminyak (Oily): {round(avg_metrics.oily or 0, 1)}/100\n"
                f"  - Rerata Flek & Noda Hitam (Dark Spots): {round(avg_metrics.dark_spots or 0, 1)}/100\n"
                f"  - Rerata Kerutan / Garis Halus: {round(avg_metrics.wrinkles or 0, 1)}/100"
            ) if avg_metrics else "  Belum ada data analisis scan."

            # 4. Customer Chat Trends (What customers ask, complain about, and request)
            cust_chat_records = db.query(ChatMessage.content, User.name, ChatMessage.created_at).join(
                User, ChatMessage.user_id == User.id
            ).filter(
                User.role == "customer",
                ChatMessage.role == "user"
            ).order_by(desc(ChatMessage.created_at)).limit(35).all()

            faq_samples = []
            complaint_samples = []
            request_samples = []
            raw_chat_samples = []

            for msg_content, u_name, c_time in cust_chat_records:
                clean_msg = msg_content.strip()
                low = clean_msg.lower()
                raw_chat_samples.append(f"  • [{c_time.strftime('%d-%m %H:%M')}] {u_name}: \"{clean_msg}\"")

                if any(w in low for w in ["resep", "obat", "bengkak", "darah", "parah", "berdarah", "kusam", "flek", "jerawat", "perih", "iritasi", "rusak", "kenapa"]):
                    complaint_samples.append(clean_msg)
                if any(w in low for w in ["rekomendasikan", "rekomendasi", "link", "shopee", "minta", "tolong", "ada berapa", "cari"]):
                    request_samples.append(clean_msg)
                if any(w in low for w in ["apa itu", "bagaimana", "apakah", "kenapa", "berapa dosis", "siapa saya"]):
                    faq_samples.append(clean_msg)

            faq_list_str = "\n".join([f"    - \"{m}\"" for m in faq_samples[:6]]) if faq_samples else "    - (Belum ada pesan pertanyaan signifikan)"
            complaints_list_str = "\n".join([f"    - \"{m}\"" for m in complaint_samples[:6]]) if complaint_samples else "    - (Belum ada keluhan khusus)"
            requests_list_str = "\n".join([f"    - \"{m}\"" for m in request_samples[:6]]) if request_samples else "    - (Belum ada permintaan khusus)"
            raw_feed_str = "\n".join(raw_chat_samples[:12]) if raw_chat_samples else "  (Belum ada riwayat obrolan customer)"

            customer_chat_section = f"""
5. ANALISIS TREN PERTANYAAN, KELUHAN & PERMINTAAN CUSTOMER (DARI TABEL CHAT_MESSAGES):
   • Total Pesan Customer Terekam: {len(cust_chat_records)} pesan terkini
   • Yang Sering Ditanyakan Customer (Frequently Asked):
{faq_list_str}
   • Yang Sering Dikeluhkan Customer (Skin Problems & Complaints):
{complaints_list_str}
   • Yang Sering Diminta Customer (Skincare & Link Requests):
{requests_list_str}
   • 12 Pesan Masuk Terakhir dari Customer:
{raw_feed_str}
"""

            # 5. Products snapshot
            total_products = db.query(Product).count()
            active_products = db.query(Product).filter(Product.status == "active").count()
            products_with_shopee = db.query(Product).filter(Product.shopee_link.isnot(None), Product.shopee_link != "").count()
            all_prods = db.query(Product).all()
            prod_summary_list = "\n".join([
                f"  • ID #{p.id} | {p.brand} {p.name} ({p.category}) | Shopee: {'Ada' if p.shopee_link else 'Belum ada'} | Masalah: {p.skin_concern or '-'} | Bahan: {p.ingredients or '-'}"
                for p in all_prods
            ])

            # 6. Activity logs snapshot
            total_logs = db.query(ActivityLog).count()
            recent_logs = db.query(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(8).all()
            log_list_str = "\n".join([
                f"  • [{l.created_at.strftime('%d-%m %H:%M')}] User #{l.user_id}: {l.activity} (IP: {l.ip_address or '-'})"
                for l in recent_logs
            ])

            # 7. System settings snapshot
            settings_records = db.query(SystemSetting).all()
            settings_str = "\n".join([
                f"  • {s.key}: {s.value}"
                for s in settings_records
                if "KEY" not in s.key.upper() and "SECRET" not in s.key.upper() and "PASSWORD" not in s.key.upper()
            ])

            # 0. Identitas admin yang sedang chat (supaya "siapa saya" dijawab
            # dari data akun yang benar, bukan dari ID scan di daftar bawah).
            identity_block = ""
            if admin_user:
                identity_block = (
                    "0. YANG SEDANG CHAT DENGANMU (ADMIN YANG LOGIN):\n"
                    f"   - Nama: {admin_user.get('name')}\n"
                    f"   - ID pengguna: #{admin_user.get('id')}  <- INI id admin yang sedang chat. "
                    "ID scan (mis. #120001) BUKAN id orang, jangan pernah dipakai sebagai id siapa pun.\n"
                    f"   - Email: {admin_user.get('email')}\n"
                    "   - Role: admin\n\n"
                )

            return f"""
DATABASE WEBSITE SKINCHECK (LIVE REPOSITORY SNAPSHOT FOR ADMIN):
{identity_block}1. TAMPILAN KARTU METRIK DASHBOARD ADMIN (REAL-TIME UI WIDGETS):
   - Total Customer: {total_customers} orang ({active_customers} Aktif)
   - Face Scans: {total_scans} pemindaian ({scans_today} hari ini)
   - Katalog Produk: {active_products} produk aktif di katalog AI
   - Analisis Berhasil (Success Rate): {success_rate} ({failed_scans} gagal)
   - AI Sessions: {total_ai_sessions} total sesi chat AI

2. TABEL USERS:
   - Total Pengguna: {total_users} (Customer: {total_customers}, Admin: {total_admins})
   - Status Akun: {active_users} Aktif, {inactive_users} Non-aktif
   - 8 Pengguna Terdaftar Terakhir:
{user_list_str}

3. TABEL SKIN_SCANS:
   - Total Pemindaian: {total_scans} (Selesai: {completed_scans}, Gagal: {failed_scans})
   - 5 Pemindaian Terakhir:
{scan_list_str}

4. TABEL ANALYSIS_RESULTS (TREN PENYAKIT & KONDISI KULIT KESELURUHAN PENGGUNA):
{avg_summary}
{customer_chat_section}
6. TABEL PRODUCTS:
   - Total Produk Katalog: {total_products} (Aktif: {active_products}, Memiliki Link Shopee: {products_with_shopee})
   - Daftar Produk di Database:
{prod_summary_list}

7. TABEL ACTIVITY_LOGS:
   - Total Log Tercatat: {total_logs}
   - 8 Log Aktivitas Terakhir:
{log_list_str}

8. TABEL SYSTEM_SETTINGS:
{settings_str}
"""
        except Exception as e:
            logger.error(f"Error building admin database context: {e}")
            return "Gagal memuat snapshot database lengkap."

    @staticmethod
    def build_product_context(db: Session, query: str) -> str:
        """Retrieves active Indonesian skincare products from the Product Catalog with Shopee links."""
        products = db.query(Product).filter(Product.status == "active").all()
        if not products:
            return "Katalog produk di database saat ini kosong."

        lines = [
            "KATALOG PRODUK RESMI DI INDONESIA (Tersimpan di Database Admin):",
            "Catatan: Gunakan HANYA data dan Link Shopee dari daftar berikut saat merekomendasikan produk:"
        ]
        
        for p in products:
            concerns = p.get_skin_concern() if p.skin_concern else []
            ingredients = p.get_ingredients() if p.ingredients else []
            shopee_str = p.shopee_link if p.shopee_link else "Belum ada link Shopee"
            lines.append(
                f"- ID: {p.id} | Brand: {p.brand} | Nama: {p.name} | Kategori: {p.category}\n"
                f"  Masalah Kulit: {', '.join(concerns) if concerns else 'Semua jenis kulit'}\n"
                f"  Bahan Aktif: {', '.join(ingredients) if ingredients else 'Tidak spesifik'}\n"
                f"  Deskripsi Singkat: {p.description or '-'}\n"
                f"  Link Shopee: {shopee_str}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 4. SYSTEM PROMPTS (ADMIN & CUSTOMER)
# ─────────────────────────────────────────────────────────

_rag_kb = RAGKnowledgeBase()

ADMIN_SYSTEM_PROMPT = """Kamu adalah Luvenna (Ven), asisten Laomera (SkinCheck). Yang lagi ngobrol sama kamu sekarang adalah ADMIN website ini — bukan customer biasa.

🎯 GAYA JAWABAN (SAMA NATURALNYA SEPERTI KE CUSTOMER):
1. NATURAL & RAMAH, BUKAN SEPERTI ROBOT: ngobrol santai layaknya rekan kerja yang paham website ini. Jangan pakai gaya laporan sistem ("Sistem aktif...", "Berdasarkan database...") atau kalimat kaku seperti formulir.
2. TO THE POINT & RINGKAS: jawab langsung apa yang ditanya, 1-3 kalimat padat (atau poin ringkas kalau memang perlu daftar). Jangan nambahin metrik lain yang nggak ditanya.
3. SAPAAN HANYA KALAU PESANNYA CUMA SAPAAN: kalau pengguna cuma bilang "hi" atau "halo", sapa balik dengan santai. Kalau dia langsung bertanya, langsung jawab tanpa pembuka.
4. ANGKA, ID, DAN NAMA HARUS DARI SNAPSHOT DI BAWAH: jangan pernah mengarang angka, ID, atau nama. Kalau datanya nggak ada di snapshot, bilang terus terang datanya nggak ketemu.
5. "SIAPA SAYA": jawab dari blok identitas admin di snapshot (nama + ID pengguna + role admin). ID scan BUKAN ID orang — jangan pernah pakai ID scan sebagai ID kamu atau ID siapa pun.
6. "SAYA BISA APA AJA": jelasin dengan santai apa aja yang bisa ditanyain ke kamu soal website ini (pengguna, scan, produk, tren, log, settings).
7. DATA SENSITIF: jangan sebar email pengguna lain kalau memang tidak ditanya soal pengguna itu. Password tidak pernah ada di konteks — jangan pernah menebak atau menyebutkannya.

🔓 YANG BISA KAMU BANTU (ADMIN PUNYA AKSES KE SELURUH DATABASE):
Kamu boleh menjawab soal SELURUH data website — bukan cuma data admin sendiri: jumlah dan daftar pengguna, pemindaian (total, selesai, gagal, terbaru, per pengguna), tren kondisi kulit semua pengguna, katalog produk, obrolan customer, log aktivitas, dan settings yang bukan rahasia. Admin juga boleh tanya apa aja soal website/data, bukan cuma soal skincare.

KONTEKS DATABASE WEBSITE LENGKAP (SNAPSHOT UNTUK ADMIN):
{admin_db_context}

RIWAYAT PERCAKAPAN SEBELUMNYA:
{chat_history}
"""

CUSTOMER_SYSTEM_PROMPT = """Nama kamu adalah Luvenna (Ven), asisten pribadi untuk kesehatan dan perawatan kulit wajah di SkinCheck.

🎯 ATURAN MUTLAK GAYA RESPONS (WAJIB DIPATUHI 100%):
1. HANYA JAWAB PERTANYAANNYA SAJA:
   - Jawab secara spesifik apa yang ditanyakan pengguna.
   - DILARANG KERAS menambahkan informasi, metrik lain, atau penjelasan di luar konteks pertanyaan!
   - Contoh: Jika pengguna bertanya tentang jerawat, HANYA jelaskan jerawatnya saja. JANGAN ceritakan minyak, kerutan, skor total, atau perawatan lain yang tidak ditanyakan.
   - Contoh: Jika pengguna bertanya definisi/fungsi kandungan (misal niacinamide), HANYA jelaskan fungsi dan cara pakainya. JANGAN tawarkan atau rekomendasikan produk jika tidak diminta!

2. TO THE POINT, RINGKAS & PADAT:
   - Sampaikan jawaban langsung ke inti persoalan tanpa berbelit-belit.
   - Cukup 1 sampai 3 kalimat padat (atau poin ringkas bila diperlukan). Hindari paragraf panjang yang melelahkan.
   - Jika ditanya "siapa kamu?", perkenalkan diri: Luvenna (Ven), asisten SkinCheck untuk kesehatan kulit wajah.
   - Jika ditanya "siapa saya?", jawab dari data nama akun dan ringkasan scannya, tanpa menyebutkan email lengkap atau password.

3. TIDAK USAH SELALU MENYAPA DI SETIAP JAWABAN:
   - JANGAN menyapa (seperti "Halo [Nama]!", "Hai!", "Selamat pagi") di awal jawaban saat menjawab pertanyaan.
   - Langsung berikan informasi intinya.
   - Sapaan HANYA diberikan jika pesan pengguna HANYA sapaan murni (seperti pengguna cuma mengetik "Halo" atau "Hai ven").

4. NATURAL & RAMAH (BUKAN SEPERTI ROBOT):
   - Gunakan gaya bahasa Indonesia yang santai, luwes, bersahabat, dan hangat layaknya teman yang paham perawatan kulit.
   - Hindari gaya kaku robot seperti "Berdasarkan sistem...", "Berdasarkan catatan database...", atau laporan formulir statistik.

5. REKOMENDASI PRODUK HANYA JIKA DIMINTA:
   - Jangan tawarkan produk atau cantumkan tautan marketplace KECUALI jika pengguna secara eksplisit meminta rekomendasi produk/skincare/link Shopee.
   - Format link Shopee jika diminta: [Beli {{Brand}} {{Nama Produk}} di Shopee]({{link}}).

6. NILAI PERSEN (%) KONDISI KULIT (JIKA DITANYAKAN):
   - Jika pengguna menanyakan kondisi kulit (jerawat, minyak, kekeringan, noda hitam, kerutan), WAJIB dijawab dalam format PERSEN (%) yang nilainya SAMA PERSIS dengan dashboard pengguna (misal "15%"). Dilarang pecahan desimal tanpa persen (seperti 0.15).

7. BATASAN MEDIS & DOMAIN:
   - Fokus pada perawatan kulit wajah non-medis. Jika di luar domain atau membutuhkan tindakan medis darurat/obat keras, tolak secara ramah dan sarankan dokter kulit spesialis.

8. JAWAB DULU, TAWARKAN SARAN SETELAHNYA:
   - Kalau pengguna bertanya tentang kondisi kulitnya (skor, kondisi dominan, hasil scan), jawab dulu pertanyaannya secara to-the-point dari data scan-nya.
   - Setelah menjawab, tawarkan saran dengan kalimat seperti "Mau aku saranin perawatannya?" — JANGAN langsung memberikan sarannya di jawaban yang sama.
   - Berikan saran perawatan atau rekomendasi produk HANYA kalau pengguna memintanya secara eksplisit (misal "kasih saran", "rekomendasi produk", "cara mengatasi").

DATA DATABASE PRIBADI PENGGUNA (HANYA UNTUK USER INI):
{user_context}

KATALOG PRODUK DARI DATABASE:
{product_context}

PENGETAHUAN PENDUKUNG (RAG KESEHATAN KULIT WAJAH):
{rag_context}

RIWAYAT PERCAKAPAN SEBELUMNYA:
{chat_history}
"""


def _get_system_prompt(db: Session, role: str) -> str:
    """Aturan jawaban AI adalah konstanta hardcode (ADMIN_SYSTEM_PROMPT /
    CUSTOMER_SYSTEM_PROMPT di file ini). Aturan TIDAK lagi dibaca dari
    database — didokumentasikan di docs/aturan-jawaban-ai.md. Parameter
    db dipertahankan supaya pemanggil tidak berubah."""
    return ADMIN_SYSTEM_PROMPT if role == "admin" else CUSTOMER_SYSTEM_PROMPT


def _render_system_prompt(db: Session, role: str, **values) -> str:
    """Render template prompt bawaan dengan str.format (jalur ini tidak
    pernah berubah satu karakter pun untuk jalur Gemini)."""
    template = _get_system_prompt(db, role)
    return template.format(**values)


# ─────────────────────────────────────────────────────────
# 5. AI ORCHESTRATOR
# ─────────────────────────────────────────────────────────

def _setting_float(db: Session, key: str, default: float) -> float:
    """Baca angka desimal dari system_settings. Nilai yang rusak (kosong,
    bukan angka) diabaikan dan diganti default, supaya chat tidak pernah
    mati gara-gara setelan yang salah ketik di halaman konfigurasi."""
    try:
        return float(SystemSetting.get_value(db, key, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _setting_int(db: Session, key: str, default: int) -> int:
    """Seperti _setting_float, untuk bilangan bulat."""
    try:
        return int(SystemSetting.get_value(db, key, str(default)))
    except (TypeError, ValueError):
        return int(default)


NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"


def _get_provider(db: Session) -> str:
    """Provider AI aktif dari database. Nilai di luar gemini/nvidia
    diperlakukan sebagai gemini (aman)."""
    p = (SystemSetting.get_value(db, "AI_PROVIDER", "gemini") or "gemini").strip().lower()
    return p if p in ("gemini", "nvidia") else "gemini"


def _call_nvidia_chat(db: Session, prompt_input: str) -> Optional[str]:
    """Satu panggilan chat ke endpoint OpenAI-compatible NVIDIA, dengan
    prompt rakitan yang SAMA PERSIS seperti yang diterima Gemini (hasil
    _build_prompt) supaya kedua provider tidak pernah menyimpang.
    Mengembalikan teks jawaban, atau None bila apa pun gagal — pemanggil
    wajib jatuh ke fallback lokal (chat tidak pernah 500 gara-gara ini)."""
    nvidia_key = (SystemSetting.get_value(db, "NVIDIA_API_KEY", "") or "").strip()
    if not nvidia_key:
        return None
    base_url = (
        SystemSetting.get_value(db, "AI_NVIDIA_BASE_URL", NVIDIA_DEFAULT_BASE_URL)
        or NVIDIA_DEFAULT_BASE_URL
    ).strip().rstrip("/") or NVIDIA_DEFAULT_BASE_URL
    model = (
        SystemSetting.get_value(db, "AI_MODEL_NVIDIA", NVIDIA_DEFAULT_MODEL)
        or NVIDIA_DEFAULT_MODEL
    ).strip() or NVIDIA_DEFAULT_MODEL
    temp_val = _setting_float(db, "AI_TEMPERATURE", 0.7)
    max_tokens_val = min(_setting_int(db, "AI_MAX_TOKENS", 4096), 1024)
    try:
        import httpx

        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {nvidia_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt_input}],
                "temperature": temp_val,
                "max_tokens": max_tokens_val,
            },
            timeout=30.0,
        )
        if resp.status_code != 200:
            logger.info(f"NVIDIA chat gagal (HTTP {resp.status_code}).")
            return None
        choices = resp.json().get("choices", [])
        if not choices:
            return None
        content = (choices[0].get("message", {}).get("content") or "").strip()
        return content or None
    except Exception as e:  # noqa: BLE001
        logger.info(f"NVIDIA chat error: {e}.")
        return None


# ─────────────────────────────────────────────────────────
# 5b. GLOSARIUM APLIKASI (pengertian konsep Laomera)
# ─────────────────────────────────────────────────────────
# (2026-11-13) Pertanyaan "apa itu ...?" soal aplikasi sebelumnya tidak
# punya tempat jatuh yang benar: cabang data menjawab dengan ANGKA ("Tidak
# ada akun yang ditangguhkan saat ini.") dan RAG hanya berisi dokumen
# SKINCARE. Glosarium ini memberi chatbot pengertian tentang seluruh
# aplikasi — ditulis dari perilaku nyata kode (bukan karangan). Isi yang
# sama didokumentasikan di docs/glosarium-aplikasi.md.
# Urutan entri = urutan pencocokan: yang spesifik dulu (multi-pose sebelum
# scan, fitur sebelum Laomera).
APP_GLOSSARY = [
    ("Akun yang ditangguhkan", ["ditangguhkan", "suspended"],
     "Akun yang ditangguhkan adalah akun yang dinonaktifkan sementara oleh admin — pemiliknya tidak bisa login dan tidak bisa memakai fitur apa pun sampai admin memulihkannya. Berbeda dengan akun pending (akun baru yang belum selesai verifikasi), akun ditangguhkan adalah akun yang sengaja diberhentikan sementara oleh admin."),
    ("Akun pending", ["pending", "menunggu verifikasi", "belum verifikasi", "belum terverifikasi"],
     "Akun pending adalah akun yang baru saja daftar tapi belum verifikasi nomor WhatsApp. Pemiliknya sudah bisa login, tapi semua fitur terkunci sampai dia memasukkan kode OTP 6 angka yang dikirim lewat WhatsApp, lalu melengkapi biodata."),
    ("Akun menunggu biodata", ["awaiting_biodata", "awaiting biodata", "menunggu biodata", "biodata belum"],
     "Akun yang sudah lolos verifikasi kode OTP tapi belum melengkapi biodata — nama, umur (5–120 tahun), dan jenis kelamin. Setelah biodatanya diisi, akunnya jadi aktif dan semua fitur terbuka."),
    ("Akun nonaktif", ["nonaktif", "inactive", "tidak aktif"],
     "Akun nonaktif adalah akun yang sedang tidak aktif dan pemiliknya tidak bisa login. Biasanya ini akun yang dinonaktifkan oleh admin — tombol hapus di panel Pengguna hanya menonaktifkan akun, bukan menghapusnya permanen, jadi admin bisa mengaktifkannya kembali."),
    ("Akun aktif", ["akun aktif"],
     "Akun aktif adalah akun yang sudah menyelesaikan semua tahap verifikasi — kode OTP WhatsApp dan biodata — dan bisa memakai semua fitur aplikasi."),
    ("Ganti nama akun", ["ganti nama", "ubah nama"],
     "Bisa — ganti nama akun lewat halaman Profil: ubah kolom nama, lalu tekan Simpan. Nama baru langsung berlaku dan tampil di dashboard serta sidebar. Yang tidak bisa diubah adalah email."),
    ("Ganti email / Gmail", ["ganti gmail", "ganti email", "ubah email", "ubah gmail", "gmail", "email"],
     "Tidak bisa — email adalah identitas untuk masuk ke akun, jadi tidak bisa diubah setelah daftar. Di halaman Profil, kolom email memang dikunci. Kalau kamu perlu pakai email lain, daftar akun baru dengan email itu."),
    ("Ganti password", ["ganti password", "ubah password", "lupa password"],
     "Customer tidak bisa ganti password sendiri — tidak ada fitur ganti password di halaman Profil. Kalau lupa password, minta admin untuk mengatur ulang passwordmu lewat halaman Pengguna. Admin bisa ganti passwordnya sendiri lewat Profil Admin, dengan memasukkan password lama (password baru minimal 6 karakter)."),
    ("Admin menonaktifkan diri sendiri", ["menonaktifkan diri", "nonaktifkan diri", "menonaktifkan akun sendiri", "nonaktifkan akun sendiri", "menonaktifkan saya", "nonaktifkan saya"],
     "Tidak bisa — admin tidak bisa menonaktifkan akunnya sendiri, supaya website tidak sampai kehilangan adminnya. Kalau akun admin perlu dinonaktifkan, harus dilakukan oleh admin lain. Aturan yang sama berlaku untuk menghapus akun admin sendiri."),
    ("Hapus akun", ["hapus akun", "hapus akun saya"],
     "Customer tidak bisa menghapus akunnya sendiri — tidak ada tombol hapus akun di halaman Profil; kalau akunmu memang harus dihapus, minta admin untuk menonaktifkannya lewat halaman Pengguna. Admin menghapus pengguna lewat tombol hapus di halaman Pengguna — itu hanya menonaktifkan akun (status jadi nonaktif), bukan menghapus permanen."),
    ("Verifikasi akun", ["verifikasi akun", "cara verifikasi", "verifikasi", "verif"],
     "Verifikasi akun ada tiga langkah: daftar (akun jadi pending), lalu masukkan kode OTP 6 angka yang dikirim ke WhatsApp (kode berlaku 10 menit dan sekali pakai), lalu isi biodata (nama, umur 5–120 tahun, jenis kelamin). Setelah itu akun jadi aktif dan semua fitur terbuka."),
    ("OTP", ["kode otp", "otp", "kode verifikasi"],
     "OTP adalah kode 6 angka yang dikirim ke nomor WhatsApp kamu saat verifikasi akun. Kodenya berlaku 10 menit, sekali pakai, dan kalau mau minta ulang harus menunggu 60 detik. Kodenya tidak pernah ditampilkan di aplikasi — cuma kamu yang bisa melihatnya di WhatsApp."),
    ("Cara chat dengan asisten", ["chat dengan asisten", "cara chat", "chat asisten", "cara ngobrol", "tombol chat"],
     "Buka tombol chat di pojok kanan bawah (ikon balon percakapan) — asisten Laomera siap menjawab soal kondisi kulitmu, produk, dan data akunmu. Riwayat chatmu tersimpan dan bisa dibuka lagi kapan pun; riwayat itu terhapus otomatis saat kamu logout."),
    ("Admin menambah produk / pengguna", ["tambah produk", "menambah produk", "tambah pengguna", "menambah pengguna", "tambah customer", "menambah customer", "buat produk", "buat pengguna", "tambah akun"],
     "Admin menambah produk lewat halaman Produk (tombol 'Tambah Produk', isi merek, nama, kategori, deskripsi, kandungan, dan tautan Shopee/Tokopedia), dan menambah pengguna lewat halaman Pengguna (tombol 'Tambah Customer', isi nama, email, dan peran)."),
    # Entri fitur admin (nama dua kata) diletakkan SEBELUM entri Scan — kalau
    # tidak, "apa itu tren scan" direbut alias "scan" yang lebih generik.
    ("AI Copilot", ["ai copilot", "copilot"],
     "AI Copilot adalah halaman admin yang membaca tren masalah kulit pengguna dan mengusulkan produk apa yang sebaiknya ditambah ke katalog — misalnya kalau rata-rata jerawat pengguna tinggi tapi produk anti-jerawat di katalog masih sedikit."),
    ("Obrolan Customer", ["obrolan customer", "obrolan"],
     "Obrolan Customer adalah halaman admin untuk memantau pertanyaan yang diajukan customer ke chatbot — admin bisa melihat pertanyaan yang paling sering ditanyakan dan membaca riwayat chat tiap customer, supaya tahu fitur apa yang membingungkan dan update apa yang dibutuhkan."),
    ("Tren Scan", ["tren scan", "tren"],
     "Tren Scan adalah halaman admin yang menampilkan rata-rata skor 5 kondisi kulit dari semua pengguna, kondisi yang paling dominan, dan rekomendasi produk untuk tiap kondisi."),
    ("Audit Scan", ["audit scan"],
     "Audit Scan adalah halaman admin yang menampilkan semua pemindaian dari semua pengguna beserta status dan hasilnya — admin bisa memeriksa scan siapa saja, kapan, dan berhasil atau gagal."),
    ("Log Aktivitas", ["log aktivitas", "log"],
     "Log Aktivitas adalah catatan semua kejadian penting di aplikasi — siapa mendaftar, siapa verifikasi, siapa scan — beserta waktunya. Admin melihatnya di halaman Log Aktivitas."),
    ("Scan multi-pose", ["multi-pose", "multi pose", "12 frame", "12 foto"],
     "Scan multi-pose adalah scan memakai kamera yang mengambil 12 foto selama 24 detik, dari 6 arah: tengah, kiri, kanan, atas, bawah, dan dekat. Setiap foto yang wajahnya terdeteksi dianalisis, lalu skor-skornya digabung jadi satu hasil."),
    ("Scan wajah", ["scan wajah", "pemindaian wajah", "pemindaian", "scan"],
     "Scan adalah pemeriksaan kondisi kulit wajah lewat foto. Ada dua cara: foto tunggal (satu foto langsung dianalisis) dan multi-pose (kamera mengambil 12 foto selama 24 detik dari 6 arah, lalu hasilnya digabung). Caranya: buka halaman Scan Kulit dari sidebar, pilih foto tunggal atau multi-pose, lalu ikuti panduan kameranya sampai selesai."),
    ("Skor 5 kondisi kulit", ["skor kondisi", "5 kondisi", "lima kondisi", "skor kulit"],
     "Ada 5 kondisi yang dinilai dari scan: jerawat, kulit kering, kulit berminyak, dark spots (noda hitam), dan kerutan. Setiap kondisi diberi skor 0–100% — ini skor KEPARAHAN: makin tinggi persentasenya, makin kuat kondisi itu terdeteksi di wajahmu."),
    ("Skor kesehatan keseluruhan", ["skor kesehatan", "overall", "skor keseluruhan"],
     "Skor kesehatan keseluruhan (overall score) adalah angka 0–100 yang merangkum hasil scanmu: 100 dikurangi rata-rata kelima skor kondisi. Makin tinggi makin sehat — di atas 75 berarti kulitmu prima, di bawah 35 berarti perlu perhatian ekstra."),
    ("Model AI VGG16", ["model ai", "vgg", "vgg16"],
     "Model AI yang menilai scanmu adalah VGG16 yang sudah dilatih khusus (file finetuned_best.h5) — model ini membaca foto wajah dan memberi skor 0–100 untuk 5 kondisi: jerawat, kulit kering, kulit berminyak, dark spots, dan kerutan. Model ini hanya menilai seberapa kuat ciri tiap kondisi terlihat di wajah — bukan mengukur luas atau lokasi jerawat."),
    ("YuNet (deteksi wajah)", ["yunet", "deteksi wajah"],
     "Sebelum dinilai, wajahmu dideteksi dulu oleh YuNet — kalau di foto tidak ada wajah yang terdeteksi, scan ditolak dan kamu diminta foto ulang. Jadi skor yang keluar selalu dihitung dari wajah, bukan dari latar foto."),
    ("Modal (AI di cloud)", ["modal"],
     "Modal adalah layanan cloud tempat model AI dijalankan — saat kamu scan, foto dikirim ke server Modal untuk dianalisis, jadi ponsel atau laptopmu tidak perlu menjalankan modelnya sendiri."),
    ("TiDB (database)", ["tidb", "database", "basis data"],
     "TiDB adalah database yang dipakai aplikasi ini di server (database MySQL yang berjalan di cloud) — semua data akun, hasil scan, produk, dan pengaturan tersimpan di sana."),
    ("Fonnte (OTP WhatsApp)", ["fonnte", "whatsapp"],
     "Fonnte adalah layanan yang mengirim kode OTP ke WhatsApp kamu saat verifikasi akun. Kalau kode OTP tidak sampai, biasanya masalahnya ada di pengaturan Fonnte (tokennya) — aplikasinya sendiri tetap jalan."),
    ("Marketplace", ["marketplace"],
     "Marketplace adalah katalog produk skincare di aplikasi — kamu bisa cari produk berdasarkan kategori (face wash, toner, serum, moisturizer, sunscreen, mask, eksfoliator) dan membelinya lewat tautan Shopee atau Tokopedia di tiap produk. Tidak ada keranjang atau pembayaran di dalam aplikasi."),
    ("Riwayat & Diagnosis", ["riwayat & diagnosis", "riwayat", "diagnosis"],
     "Riwayat & Diagnosis adalah arsip semua scan yang pernah kamu lakukan, lengkap dengan skor tiap kondisi dan perbandingan dengan scan sebelumnya — jadi kamu bisa melihat progres kesehatan kulitmu dari waktu ke waktu. Caranya: buka halaman Riwayat & Diagnosis dari sidebar — semua scanmu tampil dari yang terbaru. Panel 'Diagnosis Terkini' di atasnya menampilkan hasil scan terbarumu."),
    ("Produk", ["katalog produk", "katalog", "produk"],
     "Produk adalah item skincare di katalog aplikasi (marketplace) — masing-masing punya merek, nama, kategori, deskripsi, kandungan, dan tautan Shopee/Tokopedia. Admin mengelolanya di halaman Produk; rekomendasi produk ke customer hanya muncul kalau customer yang memintanya."),
    ("Akun admin", ["akun admin", "siapa saja admin", "daftar admin"],
     "Admin adalah pengelola aplikasi: dia bisa melihat dan mengatur pengguna, produk, audit scan, tren kulit, obrolan customer, konfigurasi, dan log aktivitas. Admin juga bisa menangguhkan atau menonaktifkan akun, dan mengubah status verifikasi pengguna."),
    ("Customer", ["customer", "pengguna"],
     "Customer adalah pengguna biasa aplikasi: dia bisa scan kulit, melihat dashboard dan riwayat diagnosisnya, belanja di marketplace, dan chat dengan asisten AI — tapi hanya soal akun dan kulitnya sendiri, bukan data pengguna lain."),
    ("Provider AI (Gemini / NVIDIA)", ["provider ai", "provider", "model ai", "gemini", "nvidia", "kunci api", "api key", "api keys"],
     "Chatbot Laomera bisa memakai dua penyedia AI: Gemini (Google) atau NVIDIA (model Llama lewat API yang kompatibel dengan OpenAI). Admin memilihnya di Konfigurasi → tab AI, mengisi kunci API-nya, dan perubahannya langsung berlaku tanpa restart. Kalau kuncinya salah atau providernya mati, chatbot tetap menjawab dari jawaban lokal, bukan error."),
    ("Konfigurasi admin", ["konfigurasi", "pengaturan", "settings"],
     "Konfigurasi adalah halaman pengaturan admin (menu Konfigurasi) dengan tiga tab: AI (provider Gemini/NVIDIA, kunci API, model, dan kuota), OTP (verifikasi WhatsApp dan aturan nomor), dan Admin (pengaturan akun admin). Perubahan di sana langsung berlaku tanpa restart."),
    ("Teknologi aplikasi", ["teknologi", "next.js", "nextjs", "fastapi", "vercel", "frontend", "backend", "dibangun dengan", "dibuat dengan"],
     "Aplikasi ini dibangun dengan frontend Next.js dan backend FastAPI, dan di-hosting di Vercel — setiap kali ada perubahan kode yang di-push ke GitHub, websitenya langsung diperbarui otomatis."),
    ("Fitur aplikasi", ["fitur"],
     "Fitur aplikasi untuk customer: Dashboard (ringkasan kondisi kulit), Scan Kulit (foto tunggal atau multi-pose 12 foto), Riwayat & Diagnosis (arsip scan + perbandingan), Marketplace (katalog produk skincare), dan chatbot Laomera Asisten. Untuk admin, tambahannya: Pengguna, Produk, Audit Scan, AI Copilot, Obrolan Customer, Tren Scan, Konfigurasi, Log Aktivitas, dan Profil Admin."),
    ("Laomera", ["laomera", "skincheck", "aplikasi ini", "aplikasi"],
     "Laomera (dulu bernama SkinCheck) adalah aplikasi web AI untuk memeriksa kesehatan kulit wajah — scan foto wajahmu, dapatkan skor 5 kondisi kulit, lalu dapatkan rekomendasi produk yang cocok."),
]

DEFINITION_TRIGGERS = [
    "apa itu", "apa yang dimaksud", "apa artinya", "apa maksud",
    "jelaskan", "maksudnya", "apa bedanya", "bedanya",
    "gimana cara", "bagaimana cara", "bagaimana saya", "gimana saya", "bagaimana",
    "apa fungsi", "fungsinya", "apa kegunaan", "untuk apa", "buat apa",
    "apakah bisa", "apakah boleh", "bolehkah", "bisakah", "bisa saya",
    "apa sih", "apa tuh", "fitur apa", "cara ",
    "teknologi apa", "apa yang dipakai", "dipakai untuk", "digunakan", "dibangun",
    "lupa password", "ganti password", "ubah password",
]

# Pertanyaan soal HASIL scan sendiri (bukan pengertian konsep) — glosarium
# tidak boleh merebutnya; biarkan jatuh ke cabang data scan.
_OWN_RESULT_MARKERS = ["hasil scan", "hasil cek", "scan terakhir", "terakhir saya", "skor saya", "skor aku"]


def _is_definition_question(msg_lower: str) -> bool:
    """Apakah pesan ini menanyakan pengertian/cara, bukan data?"""
    if any(t in msg_lower for t in DEFINITION_TRIGGERS):
        return True
    # "apakah X bisa ..." — subjek di tengah ("apakah admin bisa ...").
    return "apakah" in msg_lower and "bisa" in msg_lower


def _match_glossary(msg_lower: str):
    """Kembalikan entri glosarium yang disebut di pesan, sesuai urutan
    APP_GLOSSARY (yang spesifik dulu). List kosong = tidak ada konsep
    aplikasi yang cocok; pemanggil harus membiarkan pesan jatuh ke cabang
    berikutnya, bukan mengarang jawaban.
    Alias harus berdiri sendiri (bukan bagian kata lain): "log" di
    "teknologi" atau "produk" di "produksi" tidak cocok — tapi akhiran
    umum (-nya/-ku/-mu) tetap dihitung cocok ("menonaktifkan dirinya")."""
    matched = []
    for title, aliases, text in APP_GLOSSARY:
        for a in aliases:
            pat = r"(?<![a-z])" + re.escape(a) + r"(?:nya|ku|mu|lah|kah|pun)?(?![a-z])"
            if re.search(pat, msg_lower):
                matched.append((title, text))
                break
    return matched


def _glossary_answer(msg_lower: str):
    """Jawaban definisi untuk satu pesan, atau None kalau bukan pertanyaan
    definisi / tidak ada konsep yang cocok / yang ditanya itu hasil scan
    sendiri. Dipakai fallback admin dan customer, serta _build_prompt."""
    if not _is_definition_question(msg_lower):
        return None
    if any(w in msg_lower for w in _OWN_RESULT_MARKERS):
        return None
    matched = _match_glossary(msg_lower)
    if not matched:
        return None
    if "bedanya" in msg_lower and len(matched) >= 2:
        return "\n\n".join(f"**{t}:** {x}" for t, x in matched[:2])
    return matched[0][1]


class AIOrchestrator:
    """
    Orchestrates the entire AI Chatbot flow:
    Guardrails → Context → Memory → RAG → Gemini API → Safety Post-filter → History Persistence
    """

    def __init__(self):
        self.cached_key = None
        self.cached_model_name = None
        self.model = None

    def _save_history(self, user_id: int, user_msg: str, ai_msg: str, db: Session):
        """Saves interaction to chat_messages table for account conversation memory."""
        try:
            user_entry = ChatMessage(user_id=user_id, role="user", content=user_msg)
            ai_entry = ChatMessage(user_id=user_id, role="assistant", content=ai_msg)
            db.add(user_entry)
            db.add(ai_entry)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to persist chat memory: {e}")
            db.rollback()

    def _get_active_model(self, db: Session):
        """Dynamically loads Gemini model using database settings or fallback to .env."""
        # Di-strip: kunci/model yang ditempel dengan spasi di halaman
        # konfigurasi harus tetap bisa dipakai, bukan ditolak Google.
        api_key = (SystemSetting.get_value(db, "GEMINI_API_KEY", settings.GEMINI_API_KEY) or "").strip()
        model_name = (SystemSetting.get_value(db, "AI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"

        if not api_key:
            logger.warning("GEMINI_API_KEY is not configured.")
            return None, "API Key belum disetel"

        # Re-initialize if key or model changed
        if self.model is None or self.cached_key != api_key or self.cached_model_name != model_name:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                # Preferred model list to test (Google AI Studio active models)
                candidate_models = list(dict.fromkeys([model_name, "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite"]))
                success_model = None
                for c_model in candidate_models:
                    try:
                        m = genai.GenerativeModel(c_model)
                        success_model = m
                        model_name = c_model
                        break
                    except Exception:
                        continue

                self.model = success_model or "httpx"
                self.cached_key = api_key
                self.cached_model_name = model_name
                logger.info(f"Gemini model initialized successfully with model: {model_name}")
            except Exception as e:
                logger.info(f"Using HTTPX client for Gemini (SDK not available: {e})")
                self.model = "httpx"
                self.cached_key = api_key
                self.cached_model_name = model_name

        return self.model, None

    def _is_pure_greeting(self, message: str) -> bool:
        """Checks if user's input is strictly a greeting without any specific question."""
        cleaned = re.sub(r"[^\w\s]", "", message.lower()).strip().split()
        greeting_set = {
            "hi", "hii", "hiii", "halo", "hai", "helo", "hello", "hey", "hola",
            "p", "pagi", "siang", "sore", "malam", "assalamualaikum", "assalam",
            "ven", "luvenna", "kak", "min"
        }
        return bool(cleaned and all(w in greeting_set for w in cleaned))

    # ── Speed helpers (2026-11-04): bounded history, gated products, streaming ──

    def _recent_history(self, db: Session, user_id: int):
        """Last 8 chat messages, newest-first query bounded at the DB (the old
        code loaded the user's ENTIRE chat history, then sliced in Python)."""
        records = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == user_id)
            .order_by(desc(ChatMessage.created_at))
            .limit(8)
            .all()
        )
        records = list(reversed(records))
        lines = []
        for h in records:
            sender = "Pengguna" if h.role == "user" else "AI"
            snippet = h.content.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            lines.append(f"{sender}: {snippet}")
        text = "\n".join(lines)
        return (text if text else "Belum ada riwayat percakapan sebelumnya.", text)

    def _needs_products(self, message: str, history_raw: str) -> bool:
        """Product catalog is only sent to the model when the user is actually
        asking about products (current message or recent conversation). The old
        code attached the full catalog to EVERY prompt."""
        if self._is_seeking_product(message):
            return True
        return self._is_seeking_product(history_raw)

    def _build_prompt(self, db: Session, user_id: int, message: str, scan_id, role: str, history_str: str, rag_docs) -> str:
        """Assemble the exact prompt chat() used to build inline, so the
        streaming and non-streaming paths cannot drift apart."""
        # Glosarium ringkas (hanya kalau pertanyaannya memang definisi) supaya
        # jalur pintar (Gemini/NVIDIA) juga bisa menjawab "apa itu ...?" dari
        # pengertian yang sama dengan fallback — tanpa mengubah prompt lain.
        _glossary_hits = _match_glossary(message.lower()) if _is_definition_question(message.lower()) else []
        glossary_text = "\n".join(f"- **{t}:** {x}" for t, x in _glossary_hits) if _glossary_hits else ""
        if role == "admin":
            admin_row = db.query(User).filter(User.id == user_id).first()
            admin_info = (
                {"id": admin_row.id, "name": admin_row.name, "email": admin_row.email}
                if admin_row
                else None
            )
            admin_db_context = ContextBuilder.build_admin_database_context(
                db, message, admin_user=admin_info
            )
            system = _render_system_prompt(
                db,
                "admin",
                admin_db_context=admin_db_context,
                chat_history=history_str,
            )
            return (
                f"{system}\n\n"
                f"Instruksi Admin: \"{message}\"\n\n"
                f"Aturan gaya respons:\n"
                f"- Jawab hanya yang ditanya, to-the-point, singkat, padat, dan NATURAL seperti ngobrol santai (1-3 kalimat, atau poin ringkas kalau memang perlu daftar).\n"
                f"- Jangan menyapa di awal kecuali pesan admin cuma sapaan.\n"
                f"- Angka, ID, dan nama harus dari snapshot database di atas — jangan mengarang. "
                f"\"Siapa saya\" dijawab dari blok identitas admin di snapshot (nama + ID pengguna), bukan ID scan."
                + (
                    f"\n\nGLOSARIUM APLIKASI (kalau admin bertanya pengertian istilah/fitur, jawab dari sini, jangan mengarang):\n{glossary_text}"
                    if glossary_text
                    else ""
                )
            )
        user_context = ContextBuilder.build_user_context(user_id, db, scan_id)
        need_products = self._needs_products(message, history_str)
        if need_products:
            product_context = ContextBuilder.build_product_context(db, message)
        else:
            product_context = "Katalog produk tidak diperlukan untuk pertanyaan ini (pengguna tidak meminta rekomendasi produk)."
        rag_text = "\n".join([f"[{d.get('title', '')}]: {d.get('content', '')}" for d in rag_docs]) if rag_docs else "Tidak ada catatan khusus."
        system = _render_system_prompt(
            db,
            "customer",
            user_context=user_context,
            chat_history=history_str,
            product_context=product_context,
            rag_context=rag_text,
        )
        return (
            f"{system}\n\n"
            f"Pertanyaan Pengguna: \"{message}\"\n\n"
            f"ATURAN MUTLAK GAYA RESPONS:\n"
            f"1. HANYA JAWAB APA YANG DITANYAKAN. Dilarang keras menambahkan penjelasan atau parameter lain di luar konteks pertanyaan!\n"
            f"2. TO THE POINT, SINGKAT, PADAT, dan NATURAL (cukup 1-3 kalimat). Langsung ke inti jawaban tanpa bertele-tele.\n"
            f"3. DILARANG MENYAPA DI AWAL RESPON (jangan gunakan 'Halo [Nama]', 'Hai!', dll.) saat menjawab pertanyaan pengguna.\n"
            f"4. NADA BICARA HARUS RAMAH, NATURAL, dan HANGAT seperti teman yang paham skincare, BUKAN seperti robot kaku.\n"
            f"5. DILARANG merekomendasikan produk atau mencantumkan link pembelian KECUALI jika pengguna secara eksplisit memintanya.\n"
            f"6. Jika ditanya rerata kondisi kulit (jerawat, minyak, kekeringan, noda hitam, kerutan), WAJIB dalam format PERSEN (%) yang sinkron dengan dashboard.\n"
            f"7. Gunakan bahasa netral dan hangat, tanpa ungkapan religius spesifik."
            + (
                f"\n8. Kalau pertanyaan soal pengertian istilah/fitur aplikasi, jawab dari GLOSARIUM berikut (jangan mengarang):\n{glossary_text}"
                if glossary_text
                else ""
            )
        )

    @staticmethod
    def _chunk_sentences(text: str):
        """Split a complete fallback text into sentence chunks so even the
        non-Gemini path can be streamed to the user."""
        if not text:
            return
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        for part in parts:
            if part:
                yield part + " "

    def chat_stream(self, user_id: int, message: str, db: Session, scan_id: Optional[int] = None, role: str = "customer"):
        """Yield answer text deltas as they are generated (SSE source).

        Speed rules: bounded context (see build_user_context), product catalog
        only when asked, at most 2 model attempts, SDK streaming (no
        SDK-then-REST chain per model), 75s wall-clock budget. The full text
        is saved to history once, at the end, so the stored conversation
        matches what the user saw (post-filter applied to the saved copy).
        """
        guardrail_response = SafetyLayer.check_input(message, role=role)
        if guardrail_response:
            self._save_history(user_id, message, guardrail_response, db)
            yield guardrail_response
            return

        simulation_mode = SystemSetting.get_value(db, "AI_SIMULATION_MODE", "off")
        is_pure_greeting = self._is_pure_greeting(message)
        history_str, _history_raw = self._recent_history(db, user_id)
        rag_docs = _rag_kb.retrieve(message)

        if simulation_mode == "on":
            local_resp = SafetyLayer.post_filter(
                self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role),
                is_pure_greeting=is_pure_greeting,
            )
            self._save_history(user_id, message, local_resp, db)
            yield from self._chunk_sentences(local_resp)
            return

        # Provider NVIDIA: satu panggilan non-streaming, lalu hasilnya
        # di-chunk seperti pola REST Gemini (tanpa SDK) di bawah.
        if _get_provider(db) == "nvidia":
            prompt_input = self._build_prompt(db, user_id, message, scan_id, role, history_str, rag_docs)
            full = _call_nvidia_chat(db, prompt_input)
            if not full:
                logger.warning("NVIDIA stream tidak menjawab; fallback lokal.")
                full = self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role)
            full = SafetyLayer.post_filter(full, is_pure_greeting=is_pure_greeting)
            self._save_history(user_id, message, full, db)
            yield from self._chunk_sentences(full)
            return

        model, _err = self._get_active_model(db)
        if not model:
            local_resp = SafetyLayer.post_filter(
                self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role),
                is_pure_greeting=is_pure_greeting,
            )
            self._save_history(user_id, message, local_resp, db)
            yield from self._chunk_sentences(local_resp)
            return

        try:
            temp_val = _setting_float(db, "AI_TEMPERATURE", 0.7)
            # Cap output: the style rules demand 1-3 sentences; 4096 tokens only
            # lengthens the worst case. 1024 is still generous.
            max_tokens_val = min(_setting_int(db, "AI_MAX_TOKENS", 4096), 1024)
            current_model_name = (SystemSetting.get_value(db, "AI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
            prompt_input = self._build_prompt(db, user_id, message, scan_id, role, history_str, rag_docs)

            candidates = list(dict.fromkeys([current_model_name, "gemini-3.5-flash-lite"]))[:2]
            api_key = (self.cached_key or SystemSetting.get_value(db, "GEMINI_API_KEY", settings.GEMINI_API_KEY) or "").strip()
            full = ""
            started = False
            deadline = time.monotonic() + 75

            for m_name in candidates:
                try:
                    if self.model != "httpx":
                        import google.generativeai as genai
                        active_m = genai.GenerativeModel(m_name)
                        stream = active_m.generate_content(
                            contents=[{"role": "user", "parts": [{"text": prompt_input}]}],
                            generation_config={
                                "temperature": temp_val,
                                "max_output_tokens": max_tokens_val,
                            },
                            stream=True,
                            request_options={"timeout": 10},
                        )
                        for chunk in stream:
                            if time.monotonic() > deadline:
                                break
                            try:
                                text = chunk.text or ""
                            except Exception:
                                text = ""
                            if text:
                                started = True
                                full += text
                                yield text
                        if started:
                            break
                    else:
                        # No SDK: one REST call, then chunk it (single attempt only).
                        import httpx
                        clean_model = m_name if m_name.startswith("models/") else f"models/{m_name}"
                        url = f"https://generativelanguage.googleapis.com/v1beta/{clean_model}:generateContent?key={api_key}"
                        resp = httpx.post(
                            url,
                            json={
                                "contents": [{"parts": [{"text": prompt_input}]}],
                                "generationConfig": {
                                    "temperature": temp_val,
                                    "maxOutputTokens": max_tokens_val,
                                },
                            },
                            timeout=12.0,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            cands = data.get("candidates", [])
                            if cands:
                                parts = cands[0].get("content", {}).get("parts", [])
                                if parts and parts[0].get("text"):
                                    full = parts[0]["text"]
                                    started = True
                                    yield from self._chunk_sentences(full)
                                    break
                except Exception as model_err:
                    logger.info(f"Stream model {m_name} failed ({model_err}); {'keeping partial answer' if started else 'trying next candidate'}...")
                    if started:
                        break
                    continue

            if not full:
                logger.warning("All stream candidates failed. Falling back to local catalog matcher.")
                full = SafetyLayer.post_filter(
                    self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role),
                    is_pure_greeting=is_pure_greeting,
                )
                yield from self._chunk_sentences(full)
            else:
                full = SafetyLayer.post_filter(full, is_pure_greeting=is_pure_greeting)
            self._save_history(user_id, message, full, db)
        except Exception as e:
            logger.error(f"Gemini stream error: {e}. Falling back to local catalog matcher.")
            local_resp = SafetyLayer.post_filter(
                self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role),
                is_pure_greeting=is_pure_greeting,
            )
            self._save_history(user_id, message, local_resp, db)
            yield from self._chunk_sentences(local_resp)

    def chat(self, user_id: int, message: str, db: Session, scan_id: Optional[int] = None, role: str = "customer") -> str:
        """
        Main entry point for AI chat with persistent memory per user account and role awareness.
        """
        # 1. Guardrail check (High-risk medical & Off-topic; Admin bypasses off-topic filters)
        guardrail_response = SafetyLayer.check_input(message, role=role)
        if guardrail_response:
            self._save_history(user_id, message, guardrail_response, db)
            return guardrail_response

        # Check Simulation / Fallback Mode setting
        simulation_mode = SystemSetting.get_value(db, "AI_SIMULATION_MODE", "off")
        is_pure_greeting = self._is_pure_greeting(message)

        # 2. Build conversation memory context from database (bounded: last 8).
        chat_history_str, _history_raw = self._recent_history(db, user_id)

        # 3. Retrieve RAG documents
        rag_docs = _rag_kb.retrieve(message)

        if simulation_mode == "on":
            logger.info("AI_SIMULATION_MODE is ON. Executing smart local response generator.")
            local_resp = self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role)
            local_resp = SafetyLayer.post_filter(local_resp, is_pure_greeting=is_pure_greeting)
            self._save_history(user_id, message, local_resp, db)
            return local_resp

        # 4a. Provider NVIDIA: satu panggilan OpenAI-compatible dengan
        # prompt rakitan yang sama seperti Gemini. Gagal apa pun (kunci
        # kosong, endpoint mati, respons kosong) -> fallback lokal,
        # tidak pernah 500.
        if _get_provider(db) == "nvidia":
            prompt_input = self._build_prompt(db, user_id, message, scan_id, role, chat_history_str, rag_docs)
            ai_text = _call_nvidia_chat(db, prompt_input)
            if not ai_text:
                logger.warning("NVIDIA tidak menjawab; fallback lokal.")
                ai_text = self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role)
            ai_text = SafetyLayer.post_filter(ai_text, is_pure_greeting=is_pure_greeting)
            self._save_history(user_id, message, ai_text, db)
            return ai_text

        # 4. Get Model
        model, err = self._get_active_model(db)
        if not model:
            local_resp = self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role)
            local_resp = SafetyLayer.post_filter(local_resp, is_pure_greeting=is_pure_greeting)
            self._save_history(user_id, message, local_resp, db)
            return local_resp

        try:
            import google.generativeai as genai
            
            # 5. Read dynamic generation parameters (output capped: the style
            # rules demand 1-3 sentences, so 4096 tokens only lengthens the
            # worst case; 1024 is still generous).
            temp_val = _setting_float(db, "AI_TEMPERATURE", 0.7)
            max_tokens_val = min(_setting_int(db, "AI_MAX_TOKENS", 4096), 1024)
            current_model_name = (SystemSetting.get_value(db, "AI_MODEL", "gemini-3.5-flash-lite") or "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"

            # 6. Build prompt (shared with the streaming path; product catalog
            # only attached when the user is actually asking about products).
            prompt_input = self._build_prompt(db, user_id, message, scan_id, role, chat_history_str, rag_docs)

            # 7. Call Google Gemini: at most 2 models, SDK only. The old code
            # tried 3 models x (SDK 10s + REST 12s) = up to ~66s of serial
            # waiting before answering; the REST chain is gone. When the SDK
            # is unavailable (model == "httpx"), a single REST attempt on the
            # configured model is kept as the only fallback.
            candidate_models = list(dict.fromkeys([current_model_name, "gemini-3.5-flash-lite"]))[:2]
            if self.model == "httpx":
                candidate_models = candidate_models[:1]
            ai_text = None

            api_key = (self.cached_key or SystemSetting.get_value(db, "GEMINI_API_KEY", settings.GEMINI_API_KEY) or "").strip()
            for m_name in candidate_models:
                try:
                    # 1. Try SDK if available and not set to pure httpx
                    if self.model != "httpx":
                        try:
                            import google.generativeai as genai
                            active_m = genai.GenerativeModel(m_name)
                            response = active_m.generate_content(
                                contents=[{"role": "user", "parts": [{"text": prompt_input}]}],
                                generation_config={
                                    "temperature": temp_val,
                                    "max_output_tokens": max_tokens_val,
                                },
                                request_options={"timeout": 10}
                            )
                            if response and response.text:
                                ai_text = response.text
                                break
                        except Exception as sdk_err:
                            logger.info(f"SDK generation failed for {m_name} ({sdk_err}).")
                            if self.model != "httpx":
                                # Speed rule: no SDK-then-REST chain per model.
                                # Try the next candidate instead of stacking
                                # another 12s REST wait on a failing model.
                                continue

                    # 2. REST API fallback via httpx — only when the SDK is
                    # unavailable (model == "httpx"); then it is the sole
                    # attempt on the single configured model (see above).
                    import httpx
                    clean_model = m_name if m_name.startswith("models/") else f"models/{m_name}"
                    url = f"https://generativelanguage.googleapis.com/v1beta/{clean_model}:generateContent?key={api_key}"
                    resp = httpx.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": prompt_input}]}],
                            "generationConfig": {
                                "temperature": temp_val,
                                "maxOutputTokens": max_tokens_val,
                            }
                        },
                        timeout=12.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                ai_text = parts[0].get("text", "")
                                if ai_text:
                                    break
                except Exception as model_err:
                    logger.warning(f"Model {m_name} failed: {model_err}. Trying fallback model...")
                    continue

            if not ai_text:
                logger.warning("All Gemini candidate models returned empty or failed. Falling back to local catalog matcher.")
                ai_text = self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role)

            # 8. Post-filter safety and greetings stripping
            ai_text = SafetyLayer.post_filter(ai_text, is_pure_greeting=is_pure_greeting)
            self._save_history(user_id, message, ai_text, db)
            return ai_text

        except Exception as e:
            logger.error(f"Gemini generation error: {e}. Falling back to local catalog matcher.")
            ai_text = self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role)
            ai_text = SafetyLayer.post_filter(ai_text, is_pure_greeting=is_pure_greeting)
            self._save_history(user_id, message, ai_text, db)
            return ai_text

    def _is_seeking_product(self, message: str) -> bool:
        """Determines if the user is explicitly seeking product recommendations, active ingredients, or purchase links."""
        msg_lower = message.lower()
        product_indicators = [
            "rekomendasi", "rekomendasikan", "produk", "skincare", "beli", "link",
            "shopee", "tokopedia", "store", "toko", "merk", "brand", "cari",
            "pakai apa", "pake apa", "pilihan produk", "katalog", "jual",
            "saran produk", "rekomendasi produk", "kandungan apa yang cocok",
            "bahan apa yang cocok", "serum apa", "sunscreen apa", "moisturizer apa"
        ]
        if any(ind in msg_lower for ind in product_indicators):
            return True
        # "harga" saja (mis. "berapa harga bitcoin") bukan permintaan produk —
        # harus ada konteks produk/kulit, kalau tidak pertanyaan itu jatuh ke
        # cabang batasan domain, bukan daftar produk acak.
        if "harga" in msg_lower:
            return any(sk in msg_lower for sk in SafetyLayer.SKIN_KEYWORDS)
        return False

    def _wants_advice(self, msg_lower: str) -> bool:
        """Apakah user secara eksplisit meminta saran perawatan?
        (2026-11-13) Saran HANYA boleh muncul kalau diminta eksplisit —
        pertanyaan faktual soal data sendiri tidak boleh dijawab dengan
        saran, cukup tawaran saran sesudah menjawab."""
        return any(w in msg_lower for w in [
            "saran", "rekomendasi", "cara mengatasi", "cara ngatasin",
            "gimana ngatasin", "ngatasin", "mengatasi", "atasi",
            "pakai apa", "pake apa", "rutinitas", "hilangkan", "obati",
            "gimana cara", "bagaimana cara", "cara merawat",
            "gimana merawat", "bagaimana merawat", "tips",
        ])

    def _condition_scores(self, ar) -> dict:
        """5 skor kondisi + overall dari AnalysisResult. Fallback dark
        spots -> hyperpigmentation dan wrinkles -> texture, sama seperti
        cabang 4/5 dan tren platform."""
        return {
            "acne": int(ar.acne_score or 0),
            "dry": int(ar.dry_skin_score or 0),
            "oily": int(ar.oily_skin_score or 0),
            "dark_spots": int(ar.dark_spots_score if ar.dark_spots_score is not None and ar.dark_spots_score > 0 else (ar.hyperpigmentation_score or 0)),
            "wrinkles": int(ar.wrinkles_score if ar.wrinkles_score is not None and ar.wrinkles_score > 0 else (ar.texture_score or 0)),
            "overall": int(ar.overall_score or 0),
        }

    def _dominant_condition(self, scores: dict):
        """Kondisi dengan skor tertinggi (urutan SCORE_KEYS, strict >) —
        logika yang sama dengan 'Fokus Utama' di halaman Riwayat."""
        order = [
            ("acne", "Jerawat", "berjerawat"),
            ("dry", "Kulit kering", "kering"),
            ("oily", "Kulit berminyak", "berminyak"),
            ("dark_spots", "Dark spots", "punya flek/noda hitam"),
            ("wrinkles", "Kerutan", "punya kerutan"),
        ]
        best = None
        for key, label, phrase in order:
            v = scores.get(key, 0)
            if best is None or v > best[3]:
                best = (key, label, phrase, v)
        return best  # (key, label, phrase, value)

    def _condition_tip(self, key: str, value: float) -> str:
        """Saran perawatan per kondisi — dipakai cabang 4c dan 5.a-5.e.
        Teksnya sama dengan yang sudah ada di 5.a-5.e."""
        tips = {
            "acne": (25,
                     "Kondisi jerawatmu sudah sangat tenang dan terkontrol, pertahankan kebersihan kulit dan perlindungan tabir surya harian ya!",
                     "Untuk meredakan peradangan, Luvenna sarankan fokus menjaga skin barrier dan gunakan pembersih lembut ber-pH seimbang atau bahan aktif seperti Salicylic Acid."),
            "oily": (40,
                     "Produksi minyak wajahmu tergolong seimbang dan normal!",
                     "Kulitmu cenderung memproduksi sebum berlebih, sebaiknya pilih pelembap bertekstur gel yang ringan dan non-komedogenik agar pori-pori tetap leluasa bernapas."),
            "dry": (25,
                    "Kelembapan kulitmu terjaga dengan sangat baik dan terhidrasi prima!",
                    "Kulitmu membutuhkan hidrasi ekstra, cobalah menambahkan toner hidrasi atau pelembap berbahan ceramide dan hyaluronic acid."),
            "dark_spots": (25,
                           "Noda hitam tergolong minim, terus lindungi kulitmu dengan sunscreen setiap pagi ya!",
                           "Untuk membantu memudarkan tampilan noda hitam, kamu dapat menggunakan serum pencerah seperti Niacinamide atau Vitamin C dan disiplin mengaplikasikan tabir surya."),
            "wrinkles": (25,
                         "Kulit wajahmu terasa kenyal dan elastisitasnya terjaga prima!",
                         "Untuk merawat kekenyalan kulit dan menyamarkan garis halus, pastikan kulit selalu terhidrasi optimal dan kamu bisa mulai mempertimbangkan perawatan anti-aging ringan."),
        }
        threshold, low, high = tips[key]
        return low if value <= threshold else high

    def _latest_analysis(self, db, user_id):
        """Scan terbaru user yang sudah dianalisis + AnalysisResult-nya."""
        scans = db.query(SkinScan).filter(SkinScan.user_id == user_id).order_by(desc(SkinScan.created_at)).all()
        for s in scans:
            ar = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == s.id).first()
            if ar and ar.overall_score is not None:
                return s, ar
        return None, None

    def _is_own_skin_condition_question(self, msg_lower: str) -> bool:
        """Pertanyaan customer soal kondisi/kecenderungan kulitnya sendiri
        ('kulit saya cenderung ke apa', 'kondisi kulit saya gimana',
        'fokus perawatan saya apa'). Bukan pertanyaan soal satu kondisi
        spesifik (jerawat/minyak/dll) — itu diurus cabang 5."""
        own = any(w in msg_lower for w in ["kulit saya", "kulitku", "kulit aku", "fokus perawatan"])
        if not own:
            return False
        if any(w in msg_lower for w in ["jerawat", "acne", "minyak", "oily", "sebum", "kering", "dry", "dehidrasi", "hidrasi", "noda", "flek", "spots", "hiperpigmentasi", "kerutan", "garis halus", "wrinkles", "kerut"]):
            return False
        question_words = ["cenderung", "kondisi", "gimana", "bagaimana", "fokus", "dominan", "jenis", "ke arah", "lebih", "apa", "kenapa", "mengapa", "sehat", "skor", "berapa"]
        return any(w in msg_lower for w in question_words) or self._wants_advice(msg_lower)

    def _is_condition_knowledge_question(self, msg_lower: str) -> bool:
        """Pertanyaan PENGETAHUAN soal satu kondisi kulit ('apa penyebab
        jerawat', 'kulit saya kering banget kenapa', 'gimana cara ngilangin
        dark spots') — dijawab dokumen RAG, BUKAN skor rata-rata user
        (cabang 5). Bukan pertanyaan soal data sendiri: kalau ada kata
        skor/rata-rata/berapa/hasil scan, tetap cabang 5."""
        knowledge_words = ["penyebab", "kenapa", "mengapa", "apa itu", "ngilangin", "ngatasin", "menghilangkan", "cara menghilangkan", "apa bedanya", "faktor", "timbul", "akibat"]
        if not any(w in msg_lower for w in knowledge_words):
            return False
        if any(w in msg_lower for w in ["skor", "rata-rata", "rata rata", "rerata", "berapa", "hasil scan", "hasil cek", "persen", "persentase", "grafik", "dashboard"]):
            return False
        return any(w in msg_lower for w in ["jerawat", "acne", "minyak", "oily", "sebum", "kering", "dry", "dehidrasi", "hidrasi", "noda", "flek", "spots", "hiperpigmentasi", "kerutan", "garis halus", "wrinkles", "kerut"])

    def _generate_local_response(self, message: str, db: Session, rag_docs: List[dict], user_id: Optional[int] = None, role: str = "customer") -> str:
        """Intelligent local fallback matcher respecting role (Admin vs Customer) and to-the-point guidance."""
        msg_lower = message.lower()

        # ─────────────────────────────────────────────────────
        # ADMIN LOCAL QUERIES (Zero restrictions: Coding, DB, Customer Chat Analysis, Disease Trends)
        # ─────────────────────────────────────────────────────
        if role == "admin":
            # 0. Sapaan, identitas, dan kemampuan admin — jawaban natural
            # (bukan laporan sistem), sama santainya seperti ke customer.
            admin_u = db.query(User).filter(User.id == user_id).first() if user_id else None
            admin_name = admin_u.name if admin_u and admin_u.name else "Admin"
            cleaned_first = re.sub(r"[^\w\s]", "", msg_lower).strip().split()
            greetings = [
                "hi", "hii", "hiii", "halo", "hai", "helo", "hello", "hey", "hola",
                "p", "pagi", "siang", "sore", "malam", "assalamualaikum", "assalam",
            ]
            if cleaned_first and cleaned_first[0] in greetings:
                return (
                    f"Halo {admin_name}! Ada yang bisa dibantu? Kamu bisa tanya soal pengguna, "
                    "scan, produk, tren kulit pengguna, atau data website lainnya."
                )
            if any(w in msg_lower for w in ["siapa saya", "siapa aku", "siapa dia", "akun saya", "profil saya", "data saya", "email saya"]):
                if admin_u:
                    joined = admin_u.created_at.strftime("%d %B %Y") if admin_u.created_at else "-"
                    return (
                        f"Kamu {admin_u.name}, admin Laomera "
                        f"(email {admin_u.email}). Akunmu terdaftar sejak {joined}."
                    )
                return "Kamu login sebagai admin Laomera."
            if any(w in msg_lower for w in ["saya bisa apa", "bisa apa aja", "bisa apa saja", "apa saja yang bisa", "apa yang bisa", "kemampuan", "ngapain aja", "bisa bantu apa"]):
                return (
                    "Sebagai admin, kamu bisa tanya aku soal apa aja yang ada di database website ini — "
                    "misalnya jumlah pengguna, siapa yang paling sering scan, scan terbaru, produk di katalog, "
                    "tren masalah kulit pengguna, sampai log aktivitas. Kalau mau tanya soal perawatan kulit juga boleh."
                )

            # 0b. Definisi / pengertian konsep aplikasi (glosarium).
            # (2026-11-13) "apa itu akun yang ditangguhkan" sebelumnya dijawab
            # cabang 4b dengan JUMLAH akun ditangguhkan — padahal yang ditanya
            # itu PENGERTIAN, bukan data. Cabang ini dicek sebelum cabang data
            # (chat, tren, scan, 4b, pengguna, RAG) karena pertanyaan definisi
            # mengandung kata yang sama ("akun", "ditangguhkan", "verifikasi").
            _def_answer = _glossary_answer(msg_lower)
            if _def_answer is not None:
                return _def_answer

            # 1. Obrolan customer — tren pertanyaan + baca riwayat chat customer.
            # (2026-11-13) Daftar FAQ/keluhan/permintaan di sini sebelumnya
            # HARDCODED (data contoh), padahal query riwayat diambil tapi tidak
            # pernah dipakai. Sekarang semuanya dari ChatMessage yang sebenarnya,
            # supaya admin bisa melihat customer kebingungan di fitur apa dan
            # pertanyaan apa yang paling sering muncul (bahan update).
            if any(w in msg_lower for w in ["chat", "obrolan", "keluh", "nanya", "minta", "tren pertanyaan", "identifikasi chat", "pertanyaan customer", "pertanyaan user", "riwayat chat", "baca chat", "sering nanya", "sering ditanya"]):
                # (a) Baca riwayat chat satu customer — nama disebut di pertanyaan.
                if any(w in msg_lower for w in ["baca chat", "riwayat chat", "chatnya", "obrolan"]):
                    target = find_user_by_name(db, msg_lower, role="customer")
                    if target:
                        recent = (
                            db.query(ChatMessage)
                            .filter(ChatMessage.user_id == target.id)
                            .order_by(ChatMessage.created_at.desc())
                            .limit(8)
                            .all()
                        )
                        if not recent:
                            return f"{target.name} ({target.email}) belum pernah chat dengan asisten."
                        lines = [f"Riwayat chat {target.name} ({target.email}) — {len(recent)} pesan terakhir:"]
                        for m in reversed(recent):
                            who = "Customer" if m.role == "user" else "Asisten"
                            text = (m.content or "").strip().replace("\n", " ")
                            if len(text) > 160:
                                text = text[:160] + "…"
                            lines.append(f"• {who}: {text}")
                        return "\n".join(lines)
                # (b) Tren pertanyaan customer — data sebenarnya.
                top = top_customer_questions(db, limit=5)
                total_q = (
                    db.query(ChatMessage)
                    .join(User, ChatMessage.user_id == User.id)
                    .filter(User.role == "customer", ChatMessage.role == "user")
                    .count()
                )
                if not top:
                    return "Belum ada customer yang chat dengan asisten, jadi belum ada tren pertanyaan."
                lines = [f"Customer sudah mengajukan {total_q} pertanyaan ke asisten. Yang paling sering ditanyakan:"]
                for i, t in enumerate(top, 1):
                    askers = ", ".join(t["asked_by"][:3]) or "customer"
                    lines.append(f"{i}. \"{t['question']}\" — {t['count']} kali (oleh {askers})")
                return "\n".join(lines)

            # 2. Disease / Health Trends Query
            if any(w in msg_lower for w in ["tren penyakit", "penyakit kulit", "kondisi kulit", "masalah kulit", "tren kulit", "rerata skor", "rata-rata kulit", "masalah kulit terbanyak"]):
                avg_m = db.query(
                    func.count(AnalysisResult.id).label("total"),
                    func.avg(AnalysisResult.overall_score).label("overall"),
                    func.avg(AnalysisResult.acne_score).label("acne"),
                    func.avg(AnalysisResult.oily_skin_score).label("oily"),
                    func.avg(AnalysisResult.dry_skin_score).label("dry"),
                    func.avg(AnalysisResult.dark_spots_score).label("dark_spots"),
                    func.avg(AnalysisResult.wrinkles_score).label("wrinkles"),
                ).first()
                tot_v = getattr(avg_m, 'total', 0) or 0
                ov_v = round(getattr(avg_m, 'overall', 0) or 0, 1)
                ac_v = round(getattr(avg_m, 'acne', 0) or 0, 1)
                dr_v = round(getattr(avg_m, 'dry', 0) or 0, 1)
                oi_v = round(getattr(avg_m, 'oily', 0) or 0, 1)
                sp_v = round(getattr(avg_m, 'dark_spots', 0) or 0, 1)
                wr_v = round(getattr(avg_m, 'wrinkles', 0) or 0, 1)
                return (
                    f"**Statistik Tren Penyakit & Kondisi Kulit (Database Platform):**\n"
                    f"• Total Analisis Scan Selesai: {tot_v} data\n"
                    f"• Rerata Skor Kesehatan Keseluruhan: {ov_v}/100\n"
                    f"• Rerata Tingkat Jerawat (Acne): {ac_v}/100 (Kondisi tertinggi/paling banyak dialami)\n"
                    f"• Rerata Kulit Kering (Dry): {dr_v}/100\n"
                    f"• Rerata Kulit Berminyak (Oily): {oi_v}/100\n"
                    f"• Rerata Noda Hitam & Flek: {sp_v}/100\n"
                    f"• Rerata Kerutan / Tekstur: {wr_v}/100"
                )

            # 3. Coding / Fixing Website / General Software Engineering (Zero Restrictions for Admin)
            # NOTE: kata umum seperti "website"/"database" sengaja TIDAK masuk
            # daftar ini — kalau tidak, pertanyaan data ("aktivitas terakhir di
            # website?") malah dijawab panduan koding.
            if any(w in msg_lower for w in ["koding", "coding", "perbaiki", "fix", "bug", "python", "javascript", "fastapi", "html", "css", "controller", "fungsi", "error", "script", "ubah", "tambah", "arsitektur"]):
                return (
                    "**Panduan Teknis Sistem SkinCheck (Akses Khusus Admin):**\n"
                    "Sebagai Admin, Anda memiliki kontrol penuh atas arsitektur dan kodingan sistem. Berikut ringkasan modul utama:\n"
                    "1. **Backend & Controller (`app/controllers/`)**:\n"
                    "   - `admin_controller.py`: Manajemen pengguna, produk, log, setting sistem, dan analitik.\n"
                    "   - `customer_controller.py`: Dashboard pelanggan, alur scan wajah, cek sunscreen, riwayat, dan endpoint chat AI.\n"
                    "   - `auth_controller.py`: Otentikasi login, registrasi OTP, dan pembersihan otomatis riwayat chat saat logout.\n"
                    "2. **AI Engine & Services (`app/services/`)**:\n"
                    "   - `ai_assistant_service.py`: Orchestrator chat Luvenna, pembagian hak akses Admin (zero restrictions) vs Customer (scoped DB), RAG, dan integrasi Google Gemini.\n"
                    "   - `face_analyzer.py`: Komputasi visi computer untuk deteksi jerawat, minyak, kelembapan, flek, dan kerutan.\n"
                    "3. **Tampilan & Widget (`app/views/`)**:\n"
                    "   - `app/views/layouts/ven_assistant_widget.html`: Komponen floating AI chat.\n"
                    "   - `app/views/pages/admin/`: Panel dashboard dan manajemen data admin.\n\n"
                    "Silakan sebutkan bagian file atau fitur spesifik yang ingin dimodifikasi, dan solusi kode langsung siap diimplementasikan."
                )

            # 4. Pertanyaan soal SCAN — dicek SEBELUM snapshot pengguna.
            # (2026-11-13) "berapa total scan yang dilakukan oleh seluruh
            # pengguna" mengandung kata "pengguna", tapi yang ditanya adalah
            # scan — cabang pengguna yang dicek duluan malah menampilkan
            # data pengguna. Jadi semua pertanyaan ber-kata "scan" masuk
            # sini dulu, dan pertanyaan yang spesifik dijawab to-the-point
            # (1-2 kalimat), bukan laporan panjang.
            if any(w in msg_lower for w in ["scan", "pemindaian", "foto"]):
                # (a) Per akun — nama disebut ("berapa kali Budi scan",
                # "Budi sering scan?"). Dicek dulu supaya pertanyaan tentang
                # satu orang tidak dijawab dengan top 3 seluruh pengguna.
                named = None
                if "berapa" in msg_lower or "sering" in msg_lower:
                    named = find_user_by_name(db, msg_lower)
                if named:
                    n_total = db.query(SkinScan).filter(SkinScan.user_id == named.id).count()
                    n_ok = db.query(SkinScan).filter(SkinScan.user_id == named.id, SkinScan.status == "completed").count()
                    n_fail = db.query(SkinScan).filter(SkinScan.user_id == named.id, SkinScan.status == "failed").count()
                    return (
                        f"{named.name} sudah scan {n_total} kali — "
                        f"{n_ok} berhasil dianalisis dan {n_fail} gagal."
                    )
                # (b) Siapa yang paling banyak/sering scan — agregasi per akun, top 3.
                # (2026-11-13) "sering" saja (tanpa "paling") sebelumnya tidak
                # memicu cabang ini — "akun mana yang sering scan?" jatuh ke
                # snapshot generik yang malah menampilkan kode scan.
                if any(w in msg_lower for w in ["paling banyak", "paling sering", "tersering", "terbanyak", "paling aktif", "paling rajin", "sering", "teratas", "paling top"]):
                    top = (
                        db.query(User.name, User.email, func.count(SkinScan.id).label("cnt"))
                        .join(SkinScan, SkinScan.user_id == User.id)
                        .group_by(User.id)
                        .order_by(desc("cnt"), User.name)
                        .limit(3)
                        .all()
                    )
                    total_scans = db.query(SkinScan).count()
                    if not top:
                        return "Belum ada scan sama sekali di website ini."
                    name0, email0, cnt0 = top[0]
                    res = f"Yang paling banyak scan itu {name0} ({email0}) — {cnt0} kali dari total {total_scans} scan."
                    if len(top) > 1:
                        rest = ", ".join(f"{n} {c} kali" for n, _e, c in top[1:])
                        res += f" Di bawahnya ada {rest}."
                    return res
                # (c) Total scan keseluruhan.
                if any(w in msg_lower for w in ["total", "jumlah", "berapa"]):
                    total_scans = db.query(SkinScan).count()
                    completed = db.query(SkinScan).filter(SkinScan.status == "completed").count()
                    failed = db.query(SkinScan).filter(SkinScan.status == "failed").count()
                    return (
                        f"Total ada {total_scans} scan dari seluruh pengguna — "
                        f"{completed} berhasil dianalisis dan {failed} gagal."
                    )
                # (d) Snapshot scan generik — natural, tanpa ID scan.
                # (2026-11-13) ID scan (#240001 dst) tidak pernah diminta
                # admin; yang ditanya itu siapa + kapan + hasilnya.
                total_scans = db.query(SkinScan).count()
                completed = db.query(SkinScan).filter(SkinScan.status == "completed").count()
                failed = db.query(SkinScan).filter(SkinScan.status == "failed").count()
                recent_scans = (
                    db.query(SkinScan, User.name)
                    .join(User, SkinScan.user_id == User.id)
                    .order_by(SkinScan.created_at.desc())
                    .limit(3)
                    .all()
                )
                parts = [
                    f"Total ada {total_scans} scan — {completed} berhasil dianalisis dan {failed} gagal."
                ]
                if recent_scans:
                    s0, name0 = recent_scans[0]
                    when0 = s0.created_at.strftime("%d/%m %H:%M") if s0.created_at else "-"
                    st0 = "selesai dianalisis" if s0.status == "completed" else ("gagal dianalisis" if s0.status == "failed" else s0.status)
                    parts.append(f"Scan terbaru dari {name0}, {when0}, {st0}.")
                    if len(recent_scans) > 1:
                        others = ", ".join(
                            f"{nm} ({s.created_at.strftime('%d/%m %H:%M') if s.created_at else '-'})"
                            for s, nm in recent_scans[1:]
                        )
                        parts.append(f"Sebelumnya: {others}.")
                return " ".join(parts)

            # 4b. Status akun & verifikasi — daftar akun per status.
            # (2026-11-13) "apakah ada akun yang belum verifikasi kode OTP?"
            # jatuh ke ringkasan default karena kata "akun"/"verifikasi"
            # tidak memicu cabang mana pun (cabang pengguna hanya kenal
            # "user"/"pengguna"/"customer"/"member"). Cabang ini menjawab
            # spesifik per status, dari data User + ActivityLog yang asli.
            if any(w in msg_lower for w in ["verifikasi", "verif", "pending", "otp", "ditangguhkan", "suspended", "biodata", "akun admin", "siapa saja admin", "daftar admin", "adminnya"]):
                # (a) Daftar admin — hanya kalau pertanyaannya memang soal
                # daftar admin (bukan soal status akun admin).
                if "admin" in msg_lower and any(w in msg_lower for w in ["siapa", "daftar", "saja", "akun admin"]) and not any(w in msg_lower for w in ["belum", "verifikasi", "ditangguhkan", "suspended", "pending", "otp"]):
                    admins = db.query(User).filter(User.role == "admin").order_by(User.name).all()
                    if not admins:
                        return "Tidak ada akun admin di website ini."
                    names = ", ".join(f"{a.name} ({a.email})" for a in admins)
                    return f"Admin website ini ada {len(admins)}: {names}."
                # (b) Akun ditangguhkan.
                if any(w in msg_lower for w in ["ditangguhkan", "suspended", "tangguh"]):
                    susp = db.query(User).filter(User.status == "suspended").order_by(User.name).all()
                    if not susp:
                        return "Tidak ada akun yang ditangguhkan saat ini."
                    names = ", ".join(f"{u.name} ({u.email})" for u in susp)
                    return f"Ada {len(susp)} akun yang ditangguhkan: {names}."
                # (c) Akun belum verifikasi (pending = OTP, awaiting_biodata = biodata).
                if any(w in msg_lower for w in ["belum", "pending", "otp"]) or "biodata" in msg_lower:
                    q = db.query(User).filter(User.status.in_(["pending", "awaiting_biodata"])).order_by(User.created_at.desc())
                    biodata_only = "biodata" in msg_lower and "belum" in msg_lower
                    if biodata_only:
                        q = q.filter(User.status == "awaiting_biodata")
                    unver = q.all()
                    if not unver:
                        if biodata_only:
                            return "Tidak ada akun yang menunggu biodata saat ini."
                        return "Tidak ada akun yang belum verifikasi saat ini — semua akun sudah melewati OTP."
                    parts = []
                    for u in unver:
                        if u.status == "pending":
                            parts.append(f"{u.name} ({u.email}) masih menunggu verifikasi kode OTP")
                        else:
                            parts.append(f"{u.name} ({u.email}) sudah verifikasi OTP, belum melengkapi biodata")
                    return f"Ada {len(unver)} akun yang belum verifikasi: " + "; ".join(parts) + "."
                # (d) Akun yang baru saja terverifikasi (dari log biodata).
                if any(w in msg_lower for w in ["sudah", "baru", "terverifikasi"]):
                    logs = (
                        db.query(ActivityLog, User.name, User.email)
                        .join(User, ActivityLog.user_id == User.id)
                        .filter(ActivityLog.activity.like("%verifikasi akun selesai%"))
                        .order_by(ActivityLog.created_at.desc())
                        .limit(5)
                        .all()
                    )
                    if not logs:
                        n_active = db.query(User).filter(User.status == "active").count()
                        return (
                            "Tidak ada catatan akun yang baru saja terverifikasi di log aktivitas. "
                            f"Total ada {n_active} akun aktif saat ini."
                        )
                    names = ", ".join(
                        f"{nm} ({em}, {lg.created_at.strftime('%d/%m %H:%M') if lg.created_at else '-'})"
                        for lg, nm, em in logs
                    )
                    return f"Akun yang baru saja terverifikasi: {names}."
                # (e) Ringkasan status akun (pertanyaan status lain).
                n_active = db.query(User).filter(User.status == "active").count()
                n_pend = db.query(User).filter(User.status == "pending").count()
                n_bio = db.query(User).filter(User.status == "awaiting_biodata").count()
                n_susp = db.query(User).filter(User.status == "suspended").count()
                n_inact = db.query(User).filter(User.status == "inactive").count()
                return (
                    f"Status akun saat ini: {n_active} aktif, {n_pend} menunggu verifikasi OTP, "
                    f"{n_bio} menunggu biodata, {n_susp} ditangguhkan, dan {n_inact} nonaktif."
                )

            # 5. Users snapshot
            if any(w in msg_lower for w in ["user", "pengguna", "customer", "member", "akun"]):
                total_users = db.query(User).count()
                customers = db.query(User).filter(User.role == "customer").count()
                admins = db.query(User).filter(User.role == "admin").count()
                active = db.query(User).filter(User.status == "active").count()
                recent_users = db.query(User).order_by(User.created_at.desc()).limit(5).all()
                res = [
                    f"**Data Pengguna Website:**",
                    f"• Total Terdaftar: {total_users} orang (Customer: {customers}, Admin: {admins})",
                    f"• Status: {active} Aktif, {total_users - active} Non-aktif\n",
                    f"**5 Pengguna Terdaftar Terbaru:**"
                ]
                for u in recent_users:
                    res.append(f"• {u.name} ({u.email}) — {u.role}, {u.status}")
                return "\n".join(res)

            # 6. Products snapshot
            if any(w in msg_lower for w in ["produk", "katalog", "shopee", "barang"]):
                total_prods = db.query(Product).count()
                with_shopee = db.query(Product).filter(Product.shopee_link.isnot(None), Product.shopee_link != "").count()
                prods = db.query(Product).limit(8).all()
                res = [
                    f"**Data Katalog Produk Skincare:**",
                    f"• Total Produk: {total_prods} item",
                    f"• Memiliki Link Shopee: {with_shopee} item\n",
                    f"**Daftar Produk:**"
                ]
                for p in prods:
                    shopee_info = f" [Beli di Shopee]({p.shopee_link})" if p.shopee_link else " (Link Shopee belum ada)"
                    res.append(f"• {p.brand} {p.name} ({p.category.capitalize()}){shopee_info}")
                return "\n".join(res)

            # 7. Activity logs snapshot
            if any(w in msg_lower for w in ["log", "aktivitas", "riwayat sistem"]):
                logs = db.query(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(8).all()
                res = ["**Log Aktivitas Sistem Terkini:**"]
                for l in logs:
                    res.append(f"• [{l.created_at.strftime('%d/%m %H:%M')}] User #{l.user_id}: {l.activity} (IP: {l.ip_address or '-'})")
                return "\n".join(res)

            # 8. Admin Dashboard Cards & UI Metrics Overview
            if any(w in msg_lower for w in ["dashboard", "ringkasan", "metrik", "statistik dashboard", "angka dashboard", "tampilan admin"]):
                from datetime import date
                today = date.today()
                total_customers = db.query(User).filter(User.role == "customer").count()
                active_customers = db.query(User).filter(User.role == "customer", User.status == "active").count()
                total_scans = db.query(SkinScan).count()
                scans_today = db.query(SkinScan).filter(func.date(SkinScan.created_at) == today).count()
                active_products = db.query(Product).filter(Product.status == "active").count()
                completed_scans = db.query(SkinScan).filter(SkinScan.status == "completed").count()
                failed_scans = db.query(SkinScan).filter(SkinScan.status == "failed").count()
                total_analysis = completed_scans + failed_scans
                success_rate = f"{(completed_scans / total_analysis * 100):.1f}%" if total_analysis > 0 else "0%"
                total_ai_sessions = db.query(ChatMessage.user_id).distinct().count()

                return (
                    f"**Ringkasan Metrik Tampilan Dashboard Admin:**\n"
                    f"• **Total Customer**: {total_customers} orang ({active_customers} aktif)\n"
                    f"• **Face Scans**: {total_scans} pemindaian ({scans_today} scan hari ini)\n"
                    f"• **Katalog Produk**: {active_products} produk aktif di katalog AI\n"
                    f"• **Analisis Berhasil (Success Rate)**: {success_rate} ({failed_scans} gagal)\n"
                    f"• **AI Sessions**: {total_ai_sessions} total sesi obrolan\n"
                    f"Angka-angka di atas sama persis dengan yang tampil di kartu metrik dashboard admin."
                )

            # Default admin — pertanyaan di luar cabang data/glosarium/skincare.
            # (2026-11-13) Dulu: ringkasan angka DB untuk pertanyaan apa pun yang
            # tidak cocok — menyesatkan, karena angka itu tidak ditanya. Sekarang:
            # pertanyaan skincare dijawab dokumen RAG; sisanya dibatasi jujur,
            # tanpa angka DB dan tanpa dokumen acak.
            if rag_docs and any(sk in msg_lower for sk in SafetyLayer.SKIN_KEYWORDS):
                parts = []
                for d in rag_docs[:2]:
                    parts.append(f"**{d.get('title')}**\n{d.get('content')}")
                return "\n\n".join(parts)
            return (
                "Itu di luar yang bisa aku bantu — aku bisa jawab soal data aplikasi ini "
                "(pengguna, scan, produk, tren kulit, log aktivitas), soal pengertian "
                "istilah dan fitur di aplikasi, dan soal kesehatan kulit. Coba tanya "
                "misalnya 'berapa total pengguna' atau 'apa itu akun yang ditangguhkan'."
            )

        # ─────────────────────────────────────────────────────
        # CUSTOMER LOCAL RESPONSES (Scoped to User's Own Account & Skin Health)
        # ─────────────────────────────────────────────────────
        # 0b. Definisi / pengertian konsep aplikasi (glosarium) — sama seperti
        # admin: pertanyaan "apa itu ...?" soal aplikasi dijawab pengertiannya,
        # bukan data orang lain dan bukan dokumen acak. Ditempatkan SEBELUM
        # safety check (bukan sesudahnya) karena glosarium hanya berisi
        # pengetahuan umum aplikasi — tidak ada data pengguna di dalamnya —
        # jadi "apa itu database aplikasi" tidak boleh ditolak sebagai akses
        # tak sah. Ditempatkan sebelum cabang identitas/data supaya tidak
        # direbut cabang lain.
        _def_answer = _glossary_answer(msg_lower)
        if _def_answer is not None:
            return _def_answer

        # 1. Check if user is asking for unauthorized project-wide database or other accounts
        is_allowed_personal_db = any(ak in msg_lower for ak in [
            "siapa saya", "siapa dia", "berapa kali", "hasil cek", "rata-rata", "rata rata", "rerata",
            "link shopee", "produk shopee", "shopee yang ada di database", "shopee di database",
            "profil saya", "akun saya", "data saya", "riwayat scan", "grafik", "dashboard", "persen",
            "jerawat saya", "minyak saya", "kulit saya", "skor saya",
            "kapan terakhir", "terakhir saya scan", "scan terakhir", "status verifikasi",
            "terverifikasi", "status akun",
        ])
        if not is_allowed_personal_db:
            for pattern in SafetyLayer.UNAUTHORIZED_DB_PATTERNS:
                if re.search(pattern, msg_lower):
                    return SafetyLayer.DATABASE_RESTRICTION_REJECTION

        # 1. Identity inquiry: "siapa kamu" / "kamu siapa"
        if any(w in msg_lower for w in ["siapa kamu", "kamu siapa", "siapa namamu", "nama kamu siapa"]):
            return "Aku Luvenna (Ven), asisten SkinCheck untuk kesehatan kulit wajah."

        # 1. Greetings handling for Customer ("hi", "hiii", "halo", "hai", dll.)
        greetings = [
            "hi", "hii", "hiii", "halo", "hai", "helo", "hello", "hey", "hola",
            "p", "pagi", "siang", "sore", "malam", "assalamualaikum", "assalam",
            "halo ven", "hai ven", "ven", "luvenna"
        ]
        cleaned_first_word = re.sub(r"[^\w\s]", "", msg_lower).strip().split()
        if cleaned_first_word and cleaned_first_word[0] in greetings:
            user_name = "kamu"
            if user_id:
                u = db.query(User).filter(User.id == user_id).first()
                if u and u.name:
                    user_name = u.name
            return f"Halo {user_name}! Senang bisa menyapamu. Ada yang ingin kamu diskusikan atau tanyakan seputar kesehatan kulit wajah dan perawatanmu hari ini?"

        # 2. Customer asking who they are / profile / email ("siapa aku", "siapa saya", "profil saya", "akun saya", "email saya", "data saya")
        if user_id and any(w in msg_lower for w in ["siapa aku", "siapa saya", "siapa dia", "siapa diriku", "kamu tahu siapa", "tahu siapa aku", "tahu siapa saya", "tentang aku", "tentang saya", "akun saya", "profil saya", "data saya", "email saya", "kapan bergabung", "status akun"]):
            u = db.query(User).filter(User.id == user_id).first()
            if u:
                # Specific check if asking about email
                if any(w in msg_lower for w in ["email saya", "alamat email"]):
                    return f"Akunmu terdaftar dengan alamat email **{u.email}** atas nama **{u.name}**, dan status akunmu saat ini aktif."

                # (2026-11-13) "apakah akun saya sudah terverifikasi" sebelumnya
                # dijawab profil generik tanpa menyebut status verifikasi.
                if any(w in msg_lower for w in ["verifikasi", "verif", "terverifikasi"]):
                    verif_answers = {
                        "active": "Ya, akunmu sudah terverifikasi dan aktif — semua fitur sudah bisa kamu pakai.",
                        "pending": "Belum — akunmu masih menunggu verifikasi kode OTP WhatsApp. Cek pop-up verifikasi di dashboard untuk melanjutkan.",
                        "awaiting_biodata": "Kode OTP kamu sudah terverifikasi, tinggal lengkapi biodata supaya akunmu jadi aktif.",
                        "suspended": "Akunmu sedang ditangguhkan, jadi fitur-fiturnya belum bisa dipakai.",
                        "inactive": "Akunmu belum aktif — selesaikan verifikasi OTP dan biodata dulu.",
                    }
                    return verif_answers.get(u.status, f"Status akunmu saat ini: {u.status}.")
                if "status akun" in msg_lower or "status saya" in msg_lower:
                    status_labels = {
                        "active": "aktif",
                        "pending": "menunggu verifikasi OTP",
                        "awaiting_biodata": "menunggu biodata",
                        "suspended": "ditangguhkan",
                        "inactive": "belum aktif",
                    }
                    return f"Akunmu saat ini berstatus {status_labels.get(u.status, u.status)}."

                u_scans = db.query(SkinScan).filter(SkinScan.user_id == user_id, SkinScan.status == "completed").all()
                total_scans = len(u_scans)
                profile_data = u.get_profile_data() if hasattr(u, "get_profile_data") else {}
                concerns = profile_data.get("skin_concerns", [])
                concerns_str = ", ".join(concerns) if concerns else ""
                joined_str = u.created_at.strftime("%d %B %Y") if u.created_at else "-"

                if total_scans > 0:
                    scan_phrase = f"yang telah mempercayakan {total_scans} kali pemindaian wajah untuk memantau kesehatan kulitmu"
                else:
                    scan_phrase = "yang siap memulai langkah awal merawat kesehatan kulit wajah"

                focus_phrase = f" dengan fokus perhatian pada {concerns_str}" if concerns_str else ""

                return (
                    f"Kamu adalah **{u.name}**, pengguna terdaftar di SkinCheck sejak {joined_str}. "
                    f"Saat ini kamu {scan_phrase}{focus_phrase}."
                )

        # 3. Customer asking how many times they scanned ("berapa kali hasil cek", "berapa kali scan")
        if user_id and any(w in msg_lower for w in ["berapa kali", "total scan", "total cek", "riwayat scan", "frekuensi cek"]):
            u_scans = db.query(SkinScan).filter(SkinScan.user_id == user_id).all()
            total_s = len(u_scans)
            completed_s = len([s for s in u_scans if s.status == "completed"])
            if total_s > 0:
                return (
                    f"Sejauh ini kamu sudah mempercayakan pemeriksaan kulit wajahmu sebanyak **{total_s} kali** di SkinCheck, "
                    f"dengan {completed_s} pemindaian yang telah selesai dianalisis. Seluruh riwayatnya tersimpan rapi untuk memantau progres kesehatan kulitmu dari waktu ke waktu. "
                    f"Konsistensimu dalam memantau kondisi kulit wajah sangat baik! Mau aku saranin perawatannya?"
                )
            else:
                return "Kamu saat ini belum memiliki riwayat pemindaian wajah di SkinCheck. Luvenna siap membantumu melakukan scan wajah pertamamu kapan pun kamu siap!"

        # 4. Customer asking about Latest Scan ("scan terakhir", "hasil scan terakhir", "kondisi terakhir")
        # (2026-11-13) "kapan terakhir saya scan" sebelumnya tidak memicu
        # cabang ini (kata "terakhir ... scan" terbalik) dan jatuh ke dokumen
        # RAG yang salah ("Kapan harus ke dokter kulit?").
        if user_id and any(w in msg_lower for w in ["scan terakhir", "hasil scan terakhir", "analisis terakhir", "kondisi terakhir", "cek terakhir", "terakhir saya scan", "kapan terakhir", "terakhir scan"]):
            # (a) "kapan terakhir saya scan" — jawab tanggalnya saja, to-the-point.
            if "kapan" in msg_lower:
                latest = db.query(SkinScan).filter(SkinScan.user_id == user_id).order_by(desc(SkinScan.created_at)).first()
                if not latest:
                    return "Kamu belum pernah scan."
                when = latest.created_at.strftime("%d %B %Y, %H:%M WIB") if latest.created_at else "-"
                ar = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == latest.id).first()
                if ar and ar.overall_score is not None:
                    return f"Terakhir kamu scan tanggal {when}, skor kesehatan kulitmu {int(ar.overall_score)}/100."
                st = "selesai dianalisis" if latest.status == "completed" else ("gagal dianalisis" if latest.status == "failed" else latest.status)
                return f"Terakhir kamu scan tanggal {when} ({st})."
            u_scans = db.query(SkinScan).filter(SkinScan.user_id == user_id).order_by(desc(SkinScan.created_at)).all()
            for u_scan in u_scans:
                ar = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == u_scan.id).first()
                if ar and ar.overall_score is not None:
                    l_time = u_scan.created_at.strftime("%d %B %Y, %H:%M WIB") if u_scan.created_at else "-"
                    l_ov = int(ar.overall_score or 0)
                    l_ac = int(ar.acne_score or 0)
                    l_oi = int(ar.oily_skin_score or 0)
                    l_dr = int(ar.dry_skin_score or 0)
                    l_sp = int(ar.dark_spots_score if ar.dark_spots_score is not None and ar.dark_spots_score > 0 else (ar.hyperpigmentation_score or 0))
                    l_wr = int(ar.wrinkles_score if ar.wrinkles_score is not None and ar.wrinkles_score > 0 else (ar.texture_score or 0))

                    if l_ov >= 75:
                        summary_tone = "menunjukkan kondisi yang prima dan sangat sehat"
                    elif l_ov >= 50:
                        summary_tone = "terpantau dalam kondisi terawat dan stabil"
                    elif l_ov >= 35:
                        summary_tone = "membutuhkan asupan hidrasi dan perhatian ekstra pada perawatan harian"
                    else:
                        summary_tone = "memerlukan perawatan khusus agar barrier kulit kembali pulih"

                    return (
                        f"Pada pemindaian wajah terakhirmu tanggal {l_time}, kondisi kulitmu {summary_tone} dengan skor kesehatan **{l_ov}/100**. "
                        f"Tingkat jerawat terpantau berada di angka **{l_ac}%**, kadar minyak alami di kisaran **{l_oi}%**, dan tingkat kekeringan kulit sebesar **{l_dr}%**. "
                        f"Sementara untuk noda hitam tercatat **{l_sp}%** serta garis halus di angka **{l_wr}%**. "
                        f"Mau aku saranin perawatannya?"
                    )
            return "Belum ada riwayat hasil pemindaian wajah yang tersimpan di sistem."

        # 4c. Customer bertanya kondisi/kecenderungan kulitnya sendiri.
        # (2026-11-13) "kulit saya cenderung ke apa" sebelumnya jatuh ke
        # dokumen RAG generik ("Urutan Pemakaian Skincare Pagi" +
        # "Hyaluronic Acid"), padahal dashboard jelas menunjukkan fokus
        # utama dari scan terakhir user. Jawab dulu dari data scan user
        # (kondisi dominan + skor), baru tawarkan saran — tanpa memberi
        # sarannya di jawaban yang sama, kecuali user memang minta saran
        # secara eksplisit.
        if user_id and self._is_own_skin_condition_question(msg_lower):
            latest_scan, ar = self._latest_analysis(db, user_id)
            if not ar:
                return "Kamu belum ada scan yang berhasil dianalisis, jadi aku belum bisa baca kondisi kulitmu — mau scan dulu?"
            scores = self._condition_scores(ar)
            dom_key, dom_label, dom_phrase, dom_value = self._dominant_condition(scores)
            when = latest_scan.created_at.strftime("%d %B %Y") if latest_scan and latest_scan.created_at else "-"
            if self._wants_advice(msg_lower):
                return (
                    f"Kulitmu cenderung {dom_phrase}. Di scan terakhir tanggal {when}, "
                    f"skor {dom_label.lower()}mu {dom_value}%. "
                    f"{self._condition_tip(dom_key, dom_value)}"
                )
            score_line = (
                f"jerawat {scores['acne']}%, kulit kering {scores['dry']}%, "
                f"kulit berminyak {scores['oily']}%, dark spots {scores['dark_spots']}%, "
                f"dan kerutan {scores['wrinkles']}%"
            )
            return (
                f"Kulitmu cenderung {dom_phrase}. Di scan terakhir tanggal {when}, "
                f"skor kulitmu: {score_line}. "
                f"Mau aku saranin perawatannya?"
            )

        # 5. Customer asking for average scores / scan results / dashboard condition graphs
        has_avg_query = any(w in msg_lower for w in ["rerata", "rata-rata", "rata rata", "rata", "persen", "persentase", "grafik", "dashboard", "indeks", "skor saya", "hasil scan", "hasil cek"])
        is_acne_query = any(w in msg_lower for w in ["jerawat", "acne"])
        is_oily_query = any(w in msg_lower for w in ["minyak", "oily", "sebum"])
        is_dry_query = any(w in msg_lower for w in ["kering", "kekeringan", "dry", "dehidrasi", "hidrasi"])
        is_spots_query = any(w in msg_lower for w in ["noda", "flek", "spots", "hiperpigmentasi"])
        is_wrinkles_query = any(w in msg_lower for w in ["kerutan", "garis halus", "wrinkles", "kerut"])
        is_overall_query = any(w in msg_lower for w in ["keseluruhan", "overall", "kesehatan", "total skor"])

        # Pertanyaan pengetahuan soal kondisi (mis. "apa penyebab jerawat")
        # dijawab dokumen RAG, bukan skor rata-rata user.
        if user_id and not self._is_condition_knowledge_question(msg_lower) and (has_avg_query or any([is_acne_query, is_oily_query, is_dry_query, is_spots_query, is_wrinkles_query])):
            u_scans = db.query(SkinScan).filter(SkinScan.user_id == user_id).order_by(desc(SkinScan.created_at)).all()
            scan_ids = [s.id for s in u_scans]
            analyses = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id.in_(scan_ids)).all() if scan_ids else []
            valid_a = [a for a in analyses if a.overall_score is not None]
            analyzed_cnt = len(valid_a)

            if analyzed_cnt > 0:
                sum_ac = sum(int(a.acne_score or 0) for a in valid_a)
                sum_oi = sum(int(a.oily_skin_score or 0) for a in valid_a)
                sum_dr = sum(int(a.dry_skin_score or 0) for a in valid_a)
                sum_sp = sum(int(a.dark_spots_score if a.dark_spots_score is not None and a.dark_spots_score > 0 else (a.hyperpigmentation_score or 0)) for a in valid_a)
                sum_wr = sum(int(a.wrinkles_score if a.wrinkles_score is not None and a.wrinkles_score > 0 else (a.texture_score or 0)) for a in valid_a)
                sum_ov = sum(int(a.overall_score or 0) for a in valid_a)

                avg_ac = round(sum_ac / analyzed_cnt, 1)
                avg_oi = round(sum_oi / analyzed_cnt, 1)
                avg_dr = round(sum_dr / analyzed_cnt, 1)
                avg_sp = round(sum_sp / analyzed_cnt, 1)
                avg_wr = round(sum_wr / analyzed_cnt, 1)
                avg_ov = round(sum_ov / analyzed_cnt, 1)

                ac_status = "Perlu Treat" if avg_ac > 50 else ("Sedang" if avg_ac > 25 else "Terkontrol")
                oi_status = "Berminyak" if avg_oi > 50 else ("Normal" if avg_oi > 25 else "Kering")
                dr_status = "Dehidrasi" if avg_dr > 50 else ("Normal" if avg_dr > 25 else "Lembap")
                sp_status = "Hiperpigmentasi" if avg_sp > 40 else "Minim Noda"
                wr_status = "Perlu Anti-Aging" if avg_wr > 40 else "Elastis"

                if avg_ov >= 75:
                    ov_status = "Sangat Sehat"
                elif avg_ov >= 50:
                    ov_status = "Kondisi Terawat"
                elif avg_ov >= 35:
                    ov_status = "Perlu Hidrasi & Perawatan"
                else:
                    ov_status = "Perlu Perhatian Ekstra"

                # 5.a Specific: Acne / Jerawat
                if is_acne_query and not (is_oily_query or is_dry_query or is_spots_query or is_wrinkles_query or is_overall_query):
                    ending = self._condition_tip("acne", avg_ac) if self._wants_advice(msg_lower) else "Mau aku saranin perawatannya?"
                    return (
                        f"Tingkat jerawat pada kulit wajahmu saat ini rata-rata berada di angka **{avg_ac}%** ({ac_status}) dari riwayat {analyzed_cnt} kali pemeriksaan wajah. "
                        f"{ending}"
                    )

                # 5.b Specific: Oily / Minyak
                if is_oily_query and not (is_acne_query or is_dry_query or is_spots_query or is_wrinkles_query or is_overall_query):
                    ending = self._condition_tip("oily", avg_oi) if self._wants_advice(msg_lower) else "Mau aku saranin perawatannya?"
                    return (
                        f"Kadar minyak (sebum) alami wajahmu saat ini rata-rata berada di kisaran **{avg_oi}%** ({oi_status}) dari {analyzed_cnt} kali pemindaian wajah. "
                        f"{ending}"
                    )

                # 5.c Specific: Dry / Kekeringan
                if is_dry_query and not (is_acne_query or is_oily_query or is_spots_query or is_wrinkles_query or is_overall_query):
                    ending = self._condition_tip("dry", avg_dr) if self._wants_advice(msg_lower) else "Mau aku saranin perawatannya?"
                    return (
                        f"Tingkat kekeringan kulit wajahmu berada di rata-rata **{avg_dr}%** ({dr_status}) dari total {analyzed_cnt} kali scan wajah. "
                        f"{ending}"
                    )

                # 5.d Specific: Dark Spots / Noda Hitam
                if is_spots_query and not (is_acne_query or is_oily_query or is_dry_query or is_spots_query or is_wrinkles_query or is_overall_query):
                    ending = self._condition_tip("dark_spots", avg_sp) if self._wants_advice(msg_lower) else "Mau aku saranin perawatannya?"
                    return (
                        f"Rata-rata noda hitam atau flek di kulit wajahmu berada di angka **{avg_sp}%** ({sp_status}) dari {analyzed_cnt} riwayat scan wajah. "
                        f"{ending}"
                    )

                # 5.e Specific: Wrinkles / Garis Halus
                if is_wrinkles_query and not (is_acne_query or is_oily_query or is_dry_query or is_spots_query or is_overall_query):
                    ending = self._condition_tip("wrinkles", avg_wr) if self._wants_advice(msg_lower) else "Mau aku saranin perawatannya?"
                    return (
                        f"Rata-rata garis halus dan elastisitas kulitmu tercatat di angka **{avg_wr}%** ({wr_status}) dari {analyzed_cnt} kali pemeriksaan. "
                        f"{ending}"
                    )

                # 5.f Specific: Overall Score
                if is_overall_query and not (is_acne_query or is_oily_query or is_dry_query or is_spots_query or is_wrinkles_query):
                    return (
                        f"Indeks kesehatan kulit wajahmu secara keseluruhan berada di angka **{avg_ov}/100** ({ov_status}) dari akumulasi {analyzed_cnt} kali pemindaian wajah. "
                        f"Mau aku saranin perawatannya?"
                    )

                # 5.g Comprehensive Summary of all 5 conditions + Overall
                if has_avg_query:
                    return (
                        f"Berdasarkan riwayat {analyzed_cnt} kali pemindaian wajah yang telah kamu jalani, kondisi kulit wajahmu secara umum tergolong **{ov_status}** dengan rata-rata skor kesehatan **{avg_ov}/100**. "
                        f"Tingkat jerawatmu rata-rata berada di angka **{avg_ac}%** ({ac_status}), produksi minyak alami di kisaran **{avg_oi}%** ({oi_status}), dan kelembapan kulit terjaga dengan kekeringan terkontrol di **{avg_dr}%** ({dr_status}). "
                        f"Adapun noda hitam berada di level **{avg_sp}%** ({sp_status}) serta garis halus di angka **{avg_wr}%** ({wr_status}). "
                        f"Secara keseluruhan kulitmu terawat dengan baik, dan Luvenna siap membantu jika ada aspek khusus yang ingin kamu optimalkan!"
                    )
            else:
                return "Kamu saat ini belum memiliki riwayat pemindaian wajah yang tersimpan di sistem. Yuk coba lakukan scan wajah pertamamu!"

        # 5. Customer asking for Shopee links in database
        if any(w in msg_lower for w in ["link shopee", "produk shopee", "shopee yang ada di database", "shopee di database", "daftar shopee", "beli di shopee"]):
            prods = db.query(Product).filter(Product.status == "active", Product.shopee_link.isnot(None), Product.shopee_link != "").all()
            if prods:
                res = ["Berikut produk perawatan kulit resmi dari katalog database yang sesuai beserta tautan pembelian resminya di Shopee:"]
                for p in prods:
                    res.append(f"• **{p.brand} {p.name}** ({p.category.capitalize()}): [Beli di Shopee]({p.shopee_link})")
                return "\n".join(res)
            else:
                return "Saat ini belum ada produk dengan tautan Shopee yang tersimpan di database."

        # 6. Customer seeking product recommendation
        is_seeking = self._is_seeking_product(message)
        if is_seeking:
            all_products = db.query(Product).filter(Product.status == "active").all()
            matched = []

            keywords = {
                "sunscreen": ["sunscreen", "tabir surya", "spf", "uv"],
                "jerawat": ["jerawat", "acne", "bha", "salicylic", "komedo"],
                "serum": ["serum", "mencerahkan", "flek", "noda", "niacinamide", "vitamin c"],
                "moisturizer": ["moisturizer", "pelembap", "ceramide", "kering", "barrier"],
                "toner": ["toner", "eksfoliasi", "aha", "pha"],
                "cleanser": ["cleanser", "cuci muka", "sabun", "pembersih"]
            }

            for p in all_products:
                searchable = f"{p.brand} {p.name} {p.category} {p.skin_concern or ''} {p.ingredients or ''} {p.description or ''}".lower()
                if any(w in msg_lower for w in searchable.split() if len(w) > 3):
                    matched.append(p)
                elif any(any(kw in msg_lower for kw in kw_list) and p.category.lower() == cat for cat, kw_list in keywords.items()):
                    if p not in matched:
                        matched.append(p)

            if not matched and all_products:
                # Permintaan produk eksplisit tapi tidak ada kata kunci yang
                # cocok — tampilkan produk aktif saja, jangan jatuh ke RAG.
                matched = all_products[:3]

            if matched:
                parts = ["Berikut pilihan produk perawatan kulit wajah yang cocok dari katalog resmi:"]
                for p in matched[:3]:
                    shopee_text = f"\n  - Tautan Shopee: [Beli {p.brand} {p.name} di Shopee]({p.shopee_link})" if p.shopee_link else ""
                    parts.append(
                        f"• **{p.brand} — {p.name}** ({p.category.capitalize()})\n"
                        f"  Manfaat: {p.description or 'Membantu menjaga hidrasi dan kesehatan kulit wajah.'}\n"
                        f"  Kandungan Aktif: {p.ingredients or 'Formula teruji BPOM'}"
                        f"{shopee_text}"
                    )
                return "\n\n".join(parts)

        # 6b. Customer bertanya soal akun/scan pengguna lain — tolak (privasi).
        # (2026-11-13) "akun mana yang sering scan?" dari customer sebelumnya
        # jatuh ke dokumen RAG (jawaban skincare acak) — bukan data orang
        # lain, tapi tetap salah: pertanyaan soal akun lain harus ditolak
        # dengan jelas, bukan dijawab dokumen yang tidak nyambung.
        if any(w in msg_lower for w in ["akun mana", "siapa yang paling", "siapa yang sering", "siapa saja yang", "pengguna lain", "akun lain", "customer lain"]):
            return SafetyLayer.DATABASE_RESTRICTION_REJECTION

        # 7. Pure facial skin health / RAG inquiry
        # (2026-11-13) Dokumen RAG hanya dijawab kalau pertanyaannya memang
        # soal kulit — kalau tidak, jatuh ke cabang 8 (batasan domain), bukan
        # dokumen acak yang kebetulan cocok satu kata.
        if rag_docs and any(sk in msg_lower for sk in SafetyLayer.SKIN_KEYWORDS):
            parts = []
            for d in rag_docs[:2]:
                parts.append(f"**{d.get('title')}**\n{d.get('content')}")
            return "\n\n".join(parts)

        # 8. Domain check: If not related to skin health or personal account data, reject politely
        has_skin = any(sk in msg_lower for sk in SafetyLayer.SKIN_KEYWORDS)
        if not has_skin:
            return SafetyLayer.OFF_TOPIC_REJECTION

        return (
            "Kesehatan kulit wajah pada dasarnya bertumpu pada skin barrier yang kuat dan kebersihan pori yang terjaga. Untuk merawatnya sehari-hari:\n\n"
            "• Bersihkan wajah dengan pembersih ber-pH seimbang dua kali sehari agar kotoran terangkat tanpa membuat kulit terasa ketarik.\n"
            "• Gunakan pelembap berformula ceramide atau hyaluronic acid untuk mengunci hidrasi alami kulit.\n"
            "• Selalu aplikasikan tabir surya setiap pagi agar kulit terlindung dari radiasi UV dan terhindar dari tanda penuaan dini."
        )


# Singleton orchestrator instance
assistant = AIOrchestrator()
