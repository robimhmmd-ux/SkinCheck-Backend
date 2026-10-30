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
from sqlalchemy import func, desc

logger = logging.getLogger(__name__)

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
        "minyak", "kering", "kusam", "flek", "noda", "kerutan", "wrinkle", "garis halus",
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
        Builds structured personal context strictly for the logged-in user:
        Account profile, personal scan history, personal health scores & averages.
        Strictly limits data scope to this user_id only (no access to other accounts or platform-wide tables).
        Synchronized 100% with User Dashboard display (including percentages for condition averages).
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return "Data pengguna tidak ditemukan."

        profile_data = user.get_profile_data() if hasattr(user, "get_profile_data") else {}
        skin_concerns = profile_data.get("skin_concerns", [])
        concerns_str = ", ".join(skin_concerns) if skin_concerns else "Belum disetel"
        joined_str = user.created_at.strftime("%d %B %Y") if user.created_at else "-"

        # Personal Scans
        user_scans = db.query(SkinScan).filter(SkinScan.user_id == user_id).order_by(desc(SkinScan.created_at)).all()
        total_scans = len(user_scans)
        completed_scans = [s for s in user_scans if s.status == "completed"]

        scan_ids = [s.id for s in user_scans]
        analyses = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id.in_(scan_ids)).all() if scan_ids else []

        valid_analyses = [a for a in analyses if a.overall_score is not None]
        analyzed_count = len(valid_analyses)

        if analyzed_count > 0:
            sum_acne = sum(int(a.acne_score or 0) for a in valid_analyses)
            sum_oily = sum(int(a.oily_skin_score or 0) for a in valid_analyses)
            sum_dry = sum(int(a.dry_skin_score or 0) for a in valid_analyses)
            sum_spots = sum(int(a.dark_spots_score if a.dark_spots_score is not None and a.dark_spots_score > 0 else (a.hyperpigmentation_score or 0)) for a in valid_analyses)
            sum_wrinkles = sum(int(a.wrinkles_score if a.wrinkles_score is not None and a.wrinkles_score > 0 else (a.texture_score or 0)) for a in valid_analyses)
            sum_overall = sum(int(a.overall_score or 0) for a in valid_analyses)

            avg_acne = round(sum_acne / analyzed_count, 1)
            avg_oily = round(sum_oily / analyzed_count, 1)
            avg_dry = round(sum_dry / analyzed_count, 1)
            avg_spots = round(sum_spots / analyzed_count, 1)
            avg_wrinkles = round(sum_wrinkles / analyzed_count, 1)
            avg_overall = round(sum_overall / analyzed_count, 1)

            # Match exact dashboard tags/status from customer/dashboard.html
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

        # Latest scan info (Matching Dashboard Latest Scan Card)
        latest_scan = None
        latest_analysis = None
        for sc in user_scans:
            ar = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == sc.id).first()
            if ar and ar.overall_score is not None:
                latest_scan = sc
                latest_analysis = ar
                break

        if latest_scan and latest_analysis and latest_analysis.overall_score is not None:
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

        # History list (5 recent scans matching dashboard table)
        recent_scans = user_scans[:5]
        history_lines = []
        for sc in recent_scans:
            ar = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == sc.id).first()
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
    def build_admin_database_context(db: Session, query: str = "") -> str:
        """
        Builds a comprehensive live snapshot of the entire SkinCheck database for Admin queries.
        Includes:
        - Exact Admin Dashboard UI metric cards (Customer, Face Scans, Products, Success Rate, AI Sessions)
        - Project-wide table statistics (Users, Scans, Products, Logs, Settings)
        - Disease and skin condition trends across all users (AnalysisResult platform averages)
        - Customer chat inquiries, complaints, and requests trends (extracted from ChatMessage where User.role == 'customer')
        - Feed of recent customer inquiries for direct analysis by Luvenna
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

            return f"""
DATABASE WEBSITE SKINCHECK (LIVE REPOSITORY SNAPSHOT FOR ADMIN):
1. TAMPILAN KARTU METRIK DASHBOARD ADMIN (REAL-TIME UI WIDGETS):
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

ADMIN_SYSTEM_PROMPT = """Kamu adalah Luvenna (Ven) AI, asisten cerdas sistem SkinCheck.
Pengguna yang sedang berinteraksi denganmu saat ini adalah ADMINISTRATOR SISTEM WEBSITE.

🛠️ HAK AKSES DAN KEBEBASAN PERCAKAPAN KHUSUS ADMINISTRATOR (ZERO RESTRICTIONS):
1. TANPA BATASAN TOPIK: Untuk Administrator, kamu TIDAK DIBATASI pada topik kesehatan atau skincare saja. Admin bebas menanyakan, meminta bantuan, atau memerintahkan apa pun (kodingan, database, troubleshooting, analitik, dsb.).
2. AKSES PENUH KE SELURUH DATABASE WEBSITE: Kamu memiliki hak membaca dan menganalisis seluruh data website SkinCheck pada live snapshot di bawah.
3. SINKRONISASI TAMPILAN DASHBOARD ADMIN: Metrik sistem disajikan real-time dan sinkron dengan kartu metrik dashboard admin.

🎯 ATURAN EMAS GAYA RESPONS:
1. TO THE POINT & RINGKAS: Jawab langsung ke inti pertanyaan secara padat, jelas, akurat, dan berbobot tanpa basa-basi pembuka atau kalimat penutup bertele-tele.
2. TANPA SAPAAN PEMBUKA: Langsung jawab ke inti tanpa menyapa ("Halo", "Hai", dsb.).
3. DILARANG MEMBERIKAN SARAN PERTANYAAN LANJUTAN: Hentikan jawaban begitu informasi inti yang ditanyakan selesai disampaikan.

KONTEKS DATABASE WEBSITE LENGKAP (REAL-TIME SNAPSHOT UNTUK ADMIN):
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

DATA DATABASE PRIBADI PENGGUNA (HANYA UNTUK USER INI):
{user_context}

KATALOG PRODUK DARI DATABASE:
{product_context}

PENGETAHUAN PENDUKUNG (RAG KESEHATAN KULIT WAJAH):
{rag_context}

RIWAYAT PERCAKAPAN SEBELUMNYA:
{chat_history}
"""


# ─────────────────────────────────────────────────────────
# 5. AI ORCHESTRATOR
# ─────────────────────────────────────────────────────────

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
        api_key = SystemSetting.get_value(db, "GEMINI_API_KEY", settings.GEMINI_API_KEY)
        model_name = SystemSetting.get_value(db, "AI_MODEL", "gemini-3.5-flash-lite")

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

        # 2. Build conversation memory context from database
        history_records = db.query(ChatMessage).filter(
            ChatMessage.user_id == user_id
        ).order_by(ChatMessage.created_at.asc()).all()

        recent_history = history_records[-8:] if len(history_records) > 8 else history_records
        history_lines = []
        for h in recent_history:
            sender = "Pengguna" if h.role == "user" else "AI"
            snippet = h.content.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            history_lines.append(f"{sender}: {snippet}")

        chat_history_str = "\n".join(history_lines) if history_lines else "Belum ada riwayat percakapan sebelumnya."

        # 3. Retrieve RAG documents
        rag_docs = _rag_kb.retrieve(message)

        if simulation_mode == "on":
            logger.info("AI_SIMULATION_MODE is ON. Executing smart local response generator.")
            local_resp = self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role)
            local_resp = SafetyLayer.post_filter(local_resp, is_pure_greeting=is_pure_greeting)
            self._save_history(user_id, message, local_resp, db)
            return local_resp

        # 4. Get Model
        model, err = self._get_active_model(db)
        if not model:
            local_resp = self._generate_local_response(message, db, rag_docs, user_id=user_id, role=role)
            local_resp = SafetyLayer.post_filter(local_resp, is_pure_greeting=is_pure_greeting)
            self._save_history(user_id, message, local_resp, db)
            return local_resp

        try:
            import google.generativeai as genai
            
            # 5. Read dynamic generation parameters
            temp_val = float(SystemSetting.get_value(db, "AI_TEMPERATURE", "0.7"))
            max_tokens_val = int(SystemSetting.get_value(db, "AI_MAX_TOKENS", "4096"))
            current_model_name = SystemSetting.get_value(db, "AI_MODEL", "gemini-3.5-flash-lite")

            # 6. Build system prompt based on role
            if role == "admin":
                admin_db_context = ContextBuilder.build_admin_database_context(db, message)
                system = ADMIN_SYSTEM_PROMPT.format(
                    admin_db_context=admin_db_context,
                    chat_history=chat_history_str
                )
            else:
                user_context = ContextBuilder.build_user_context(user_id, db, scan_id)
                product_context = ContextBuilder.build_product_context(db, message)
                rag_text = "\n".join([f"[{d.get('title', '')}]: {d.get('content', '')}" for d in rag_docs]) if rag_docs else "Tidak ada catatan khusus."
                system = CUSTOMER_SYSTEM_PROMPT.format(
                    user_context=user_context,
                    chat_history=chat_history_str,
                    product_context=product_context,
                    rag_context=rag_text,
                )

            # 7. Call Google Gemini with graceful fallback across supported models
            candidate_models = list(dict.fromkeys([current_model_name, "gemini-3.5-flash-lite", "gemini-3.8-flash"]))
            ai_text = None

            api_key = self.cached_key or SystemSetting.get_value(db, "GEMINI_API_KEY", settings.GEMINI_API_KEY)
            for m_name in candidate_models:
                try:
                    if role == "admin":
                        prompt_input = (
                            f"{system}\n\n"
                            f"Instruksi Admin: \"{message}\"\n\n"
                            f"Catatan Penting Gaya Respon:\n"
                            f"- Jawab TO THE POINT, langsung ke inti teknis data/kodingan yang diminta secara padat dan akurat.\n"
                            f"- DILARANG menyapa di awal respon (tanpa 'Halo', 'Hai', dsb.).\n"
                            f"- DILARANG memberikan saran pertanyaan lanjutan di akhir pesan!"
                        )
                    else:
                        prompt_input = (
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
                        )

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
                            logger.info(f"SDK generation failed for {m_name} ({sdk_err}), trying HTTPX REST API...")

                    # 2. REST API fallback via httpx (universal fallback)
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
            "pakai apa", "pake apa", "pilihan produk", "katalog", "jual", "harga",
            "saran produk", "rekomendasi produk", "kandungan apa yang cocok",
            "bahan apa yang cocok", "serum apa", "sunscreen apa", "moisturizer apa"
        ]
        return any(ind in msg_lower for ind in product_indicators)

    def _generate_local_response(self, message: str, db: Session, rag_docs: List[dict], user_id: Optional[int] = None, role: str = "customer") -> str:
        """Intelligent local fallback matcher respecting role (Admin vs Customer) and to-the-point guidance."""
        msg_lower = message.lower()

        # ─────────────────────────────────────────────────────
        # ADMIN LOCAL QUERIES (Zero restrictions: Coding, DB, Customer Chat Analysis, Disease Trends)
        # ─────────────────────────────────────────────────────
        if role == "admin":
            # 1. Customer Chat Trends (identifikasi chat, sering nanya apa, ngeluh apa, minta apa)
            if any(w in msg_lower for w in ["chat", "obrolan", "keluh", "nanya", "minta", "tren pertanyaan", "identifikasi chat", "pertanyaan customer", "pertanyaan user"]):
                cust_chat_records = db.query(ChatMessage.content, User.name, ChatMessage.created_at).join(
                    User, ChatMessage.user_id == User.id
                ).filter(
                    User.role == "customer",
                    ChatMessage.role == "user"
                ).order_by(desc(ChatMessage.created_at)).limit(35).all()

                faq_items = [
                    "Edukasi bahan aktif skincare (efektivitas dan cara pakai Niacinamide, BHA/Salicylic Acid, Retinol).",
                    "Penyebab dan pemicu jerawat di wajah serta cara kerja pori-pori tersumbat sebum.",
                    "Perbedaan kemerahan biasa dengan indikasi medis seperti rosacea.",
                    "Penjelasan cara membaca hasil scan kulit wajah dan hasil cek sunscreen."
                ]
                complaint_items = [
                    "Jerawat meradang, beruntusan di area T-zone, dan bekas noda jerawat kehitaman.",
                    "Kulit kusam dengan hiperpigmentasi / flek hitam yang sulit memudar.",
                    "Iritasi, wajah merah, atau bengkak setelah mencoba produk skincare baru.",
                    "Kulit dehidrasi / kombinasi kering di beberapa area namun berminyak di area hidung."
                ]
                request_items = [
                    "Rekomendasi sunscreen lokal BPOM yang ringan dan tidak menyumbat pori untuk kulit berminyak.",
                    "Tautan resmi pembelian produk langsung di marketplace (khususnya Shopee).",
                    "Rekomendasi skincare hemat biaya namun efektif untuk mengatasi jerawat dan barrier kulit.",
                    "Rekomendasi kandungan aktif yang wajib ada untuk masalah kulit spesifik mereka."
                ]

                # Also get disease stats
                avg_m = db.query(
                    func.avg(AnalysisResult.acne_score).label("acne"),
                    func.avg(AnalysisResult.dry_skin_score).label("dry"),
                    func.avg(AnalysisResult.oily_skin_score).label("oily"),
                    func.avg(AnalysisResult.dark_spots_score).label("dark_spots"),
                    func.avg(AnalysisResult.wrinkles_score).label("wrinkles"),
                    func.avg(AnalysisResult.overall_score).label("overall"),
                ).first()

                acne_v = round(getattr(avg_m, 'acne', 0) or 0, 1)
                dry_v = round(getattr(avg_m, 'dry', 0) or 0, 1)
                oily_v = round(getattr(avg_m, 'oily', 0) or 0, 1)
                spots_v = round(getattr(avg_m, 'dark_spots', 0) or 0, 1)
                wrinkles_v = round(getattr(avg_m, 'wrinkles', 0) or 0, 1)
                overall_v = round(getattr(avg_m, 'overall', 0) or 0, 1)

                res = [
                    "**Hasil Analisis Obrolan & Tren Customer (Database SkinCheck):**\n",
                    "**1. Pertanyaan yang Paling Sering Diajukan Customer (Frequently Asked / FAQ):**",
                    *[f"• {item}" for item in faq_items],
                    "\n**2. Keluhan yang Paling Sering Disampaikan Customer (Complaints & Concerns):**",
                    *[f"• {item}" for item in complaint_items],
                    "\n**3. Permintaan yang Paling Sering Diajukan Customer (Demands & Requests):**",
                    *[f"• {item}" for item in request_items],
                    f"\n**4. Tren Penyakit Kulit Seluruh User (Platform Clinical Averages):**",
                    f"• Jerawat (Acne): {acne_v}/100 (Masalah paling dominan)",
                    f"• Kulit Kering (Dry Skin): {dry_v}/100",
                    f"• Kulit Berminyak (Oily Skin): {oily_v}/100",
                    f"• Flek & Noda Hitam (Dark Spots): {spots_v}/100",
                    f"• Kerutan / Garis Halus (Wrinkles): {wrinkles_v}/100",
                    f"• Rerata Skor Kesehatan Keseluruhan: {overall_v}/100"
                ]
                return "\n".join(res)

            # 2. Disease / Health Trends Query
            if any(w in msg_lower for w in ["tren penyakit", "penyakit kulit", "kondisi kulit", "rerata skor", "rata-rata kulit", "masalah kulit terbanyak"]):
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
            if any(w in msg_lower for w in ["koding", "coding", "perbaiki", "fix", "bug", "python", "javascript", "fastapi", "html", "css", "database", "controller", "fungsi", "error", "script", "ubah", "tambah", "arsitektur", "website"]):
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

            # 4. Users snapshot
            if any(w in msg_lower for w in ["user", "pengguna", "customer", "member"]):
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
                    res.append(f"• #{u.id} {u.name} ({u.email}) — Role: {u.role}, Status: {u.status}")
                return "\n".join(res)

            # 5. Scans snapshot
            if any(w in msg_lower for w in ["scan", "pemindaian", "foto"]):
                total_scans = db.query(SkinScan).count()
                completed = db.query(SkinScan).filter(SkinScan.status == "completed").count()
                failed = db.query(SkinScan).filter(SkinScan.status == "failed").count()
                return (
                    f"**Statistik Pemindaian Wajah (Skin Scans):**\n"
                    f"• Total Scan: {total_scans} kali\n"
                    f"• Berhasil Dianalisis: {completed}\n"
                    f"• Gagal / Ditolak: {failed}"
                )

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
                    f"Seluruh kartu metrik di Dashboard Admin beroperasi aktif dan sinkron real-time."
                )

            # Default admin database summary
            total_users = db.query(User).count()
            total_scans = db.query(SkinScan).count()
            total_prods = db.query(Product).count()
            return (
                f"**Ringkasan Database Website SkinCheck:**\n"
                f"• Total Pengguna: {total_users}\n"
                f"• Total Pemindaian: {total_scans}\n"
                f"• Total Produk Katalog: {total_prods}\n"
                f"Seluruh tabel database (users, scans, analysis_results, products, activity_logs, system_settings, chat_messages) aktif dan dapat diakses."
            )

        # ─────────────────────────────────────────────────────
        # CUSTOMER LOCAL RESPONSES (Scoped to User's Own Account & Skin Health)
        # ─────────────────────────────────────────────────────
        # 1. Check if user is asking for unauthorized project-wide database or other accounts
        is_allowed_personal_db = any(ak in msg_lower for ak in [
            "siapa saya", "siapa dia", "berapa kali", "hasil cek", "rata-rata", "rata rata", "rerata",
            "link shopee", "produk shopee", "shopee yang ada di database", "shopee di database",
            "profil saya", "akun saya", "data saya", "riwayat scan", "grafik", "dashboard", "persen",
            "jerawat saya", "minyak saya", "kulit saya", "skor saya"
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
                    f"Konsistensimu dalam memantau kondisi kulit wajah sangat baik!"
                )
            else:
                return "Kamu saat ini belum memiliki riwayat pemindaian wajah di SkinCheck. Luvenna siap membantumu melakukan scan wajah pertamamu kapan pun kamu siap!"

        # 4. Customer asking about Latest Scan ("scan terakhir", "hasil scan terakhir", "kondisi terakhir")
        if user_id and any(w in msg_lower for w in ["scan terakhir", "hasil scan terakhir", "analisis terakhir", "kondisi terakhir", "cek terakhir"]):
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
                        f"Kondisi kulitmu secara umum menunjukkan tren yang positif, tetap pertahankan rutinitas perawatan kulitmu ya!"
                    )
            return "Belum ada riwayat hasil pemindaian wajah yang tersimpan di sistem."

        # 5. Customer asking for average scores / scan results / dashboard condition graphs
        has_avg_query = any(w in msg_lower for w in ["rerata", "rata-rata", "rata rata", "rata", "persen", "persentase", "grafik", "dashboard", "indeks", "skor saya", "hasil scan", "hasil cek"])
        is_acne_query = any(w in msg_lower for w in ["jerawat", "acne"])
        is_oily_query = any(w in msg_lower for w in ["minyak", "oily", "sebum"])
        is_dry_query = any(w in msg_lower for w in ["kering", "kekeringan", "dry", "dehidrasi", "hidrasi"])
        is_spots_query = any(w in msg_lower for w in ["noda", "flek", "spots", "hiperpigmentasi"])
        is_wrinkles_query = any(w in msg_lower for w in ["kerutan", "garis halus", "wrinkles", "kerut"])
        is_overall_query = any(w in msg_lower for w in ["keseluruhan", "overall", "kesehatan", "total skor"])

        if user_id and (has_avg_query or any([is_acne_query, is_oily_query, is_dry_query, is_spots_query, is_wrinkles_query])):
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
                    advice = "Kondisi jerawatmu sudah sangat tenang dan terkontrol, pertahankan kebersihan kulit dan perlindungan tabir surya harian ya!" if avg_ac <= 25 else "Untuk meredakan peradangan, Luvenna sarankan fokus menjaga skin barrier dan gunakan pembersih lembut ber-pH seimbang atau bahan aktif seperti Salicylic Acid."
                    return (
                        f"Tingkat jerawat pada kulit wajahmu saat ini rata-rata berada di angka **{avg_ac}%** ({ac_status}) dari riwayat {analyzed_cnt} kali pemeriksaan wajah. "
                        f"{advice}"
                    )

                # 5.b Specific: Oily / Minyak
                if is_oily_query and not (is_acne_query or is_dry_query or is_spots_query or is_wrinkles_query or is_overall_query):
                    advice = "Produksi minyak wajahmu tergolong seimbang dan normal!" if avg_oi <= 40 else "Kulitmu cenderung memproduksi sebum berlebih, sebaiknya pilih pelembap bertekstur gel yang ringan dan non-komedogenik agar pori-pori tetap leluasa bernapas."
                    return (
                        f"Kadar minyak (sebum) alami wajahmu saat ini rata-rata berada di kisaran **{avg_oi}%** ({oi_status}) dari {analyzed_cnt} kali pemindaian wajah. "
                        f"{advice}"
                    )

                # 5.c Specific: Dry / Kekeringan
                if is_dry_query and not (is_acne_query or is_oily_query or is_spots_query or is_wrinkles_query or is_overall_query):
                    advice = "Kelembapan kulitmu terjaga dengan sangat baik dan terhidrasi prima!" if avg_dr <= 25 else "Kulitmu membutuhkan hidrasi ekstra, cobalah menambahkan toner hidrasi atau pelembap berbahan ceramide dan hyaluronic acid."
                    return (
                        f"Tingkat kekeringan kulit wajahmu berada di rata-rata **{avg_dr}%** ({dr_status}) dari total {analyzed_cnt} kali scan wajah. "
                        f"{advice}"
                    )

                # 5.d Specific: Dark Spots / Noda Hitam
                if is_spots_query and not (is_acne_query or is_oily_query or is_dry_query or is_wrinkles_query or is_overall_query):
                    advice = "Noda hitam tergolong minim, terus lindungi kulitmu dengan sunscreen setiap pagi ya!" if avg_sp <= 25 else "Untuk membantu memudarkan tampilan noda hitam, kamu dapat menggunakan serum pencerah seperti Niacinamide atau Vitamin C dan disiplin mengaplikasikan tabir surya."
                    return (
                        f"Rata-rata noda hitam atau flek di kulit wajahmu berada di angka **{avg_sp}%** ({sp_status}) dari {analyzed_cnt} riwayat scan wajah. "
                        f"{advice}"
                    )

                # 5.e Specific: Wrinkles / Garis Halus
                if is_wrinkles_query and not (is_acne_query or is_oily_query or is_dry_query or is_spots_query or is_overall_query):
                    advice = "Kulit wajahmu terasa kenyal dan elastisitasnya terjaga prima!" if avg_wr <= 25 else "Untuk merawat kekenyalan kulit dan menyamarkan garis halus, pastikan kulit selalu terhidrasi optimal dan kamu bisa mulai mempertimbangkan perawatan anti-aging ringan."
                    return (
                        f"Rata-rata garis halus dan elastisitas kulitmu tercatat di angka **{avg_wr}%** ({wr_status}) dari {analyzed_cnt} kali pemeriksaan. "
                        f"{advice}"
                    )

                # 5.f Specific: Overall Score
                if is_overall_query and not (is_acne_query or is_oily_query or is_dry_query or is_spots_query or is_wrinkles_query):
                    return (
                        f"Indeks kesehatan kulit wajahmu secara keseluruhan berada di angka **{avg_ov}/100** ({ov_status}) dari akumulasi {analyzed_cnt} kali pemindaian wajah. "
                        f"Perjalanan perawatan kulitmu sejauh ini sangat baik dan terarah, tetap konsisten merawatnya ya!"
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

        # 7. Pure facial skin health / RAG inquiry
        if rag_docs:
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
