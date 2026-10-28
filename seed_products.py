from app.config.database import SessionLocal
from app.models.product import Product
import json

db = SessionLocal()

sample_products = [
    {
        "brand": "SkinCheck",
        "name": "Gentle Facial Cleanser",
        "category": "cleanser",
        "concerns": ["jerawat", "berminyak"],
        "desc": "Pembersih wajah pH balanced bebas sulfat untuk kulit sensitif dan rentan berjerawat.",
        "usage": "Gunakan pagi dan malam hari pada wajah basah, bilas hingga bersih.",
        "ingredients": "Water, Glycerin, Sodium Cocoyl Isethionate, Centella Asiatica Extract, Tea Tree Leaf Oil"
    },
    {
        "brand": "SkinCheck",
        "name": "Niacinamide 5% Serum",
        "category": "serum",
        "concerns": ["kemerahan", "dark spots", "bekas jerawat"],
        "desc": "Serum pencerah dan pelindung skin barrier dengan Niacinamide murni dan Zinc PCA.",
        "usage": "Teteskan 2-3 tetes pada wajah setelah toner, ratakan perlahan.",
        "ingredients": "Niacinamide, Zinc PCA, Hyaluronic Acid, Centella Asiatica, Allantoin"
    },
    {
        "brand": "SkinCheck",
        "name": "Daily Sunscreen SPF 50 PA++++",
        "category": "sunscreen",
        "concerns": ["semua", "anti-aging", "sun protection"],
        "desc": "Tabir surya ringan tanpa whitecast dengan perlindungan broad-spectrum UV.",
        "usage": "Aplikasikan sebanyak dua jari 15 menit sebelum terpapar sinar matahari.",
        "ingredients": "Zinc Oxide, Ethylhexyl Methoxycinnamate, Niacinamide, Tocopherol, Ceramide NP"
    },
    {
        "brand": "SkinCheck",
        "name": "Hydrating Ceramide Moisturizer",
        "category": "moisturizer",
        "concerns": ["kering", "kulit mengelupas", "wrinkles"],
        "desc": "Pelembap restorasi barrier kulit dengan 5x Ceramides dan Hyaluronic Acid.",
        "usage": "Oleskan secara merata pada wajah dan leher sebagai langkah akhir rutinitas.",
        "ingredients": "Ceramide NP, Ceramide AP, Ceramide EOP, Hyaluronic Acid, Squalane, Shea Butter"
    },
    {
        "brand": "SkinCheck",
        "name": "BHA Clarifying Acne Toner",
        "category": "toner",
        "concerns": ["jerawat", "komedo", "pori-pori besar"],
        "desc": "Toner eksfoliasi lembut dengan Salicylic Acid 2% untuk membersihkan pori tersumbat.",
        "usage": "Gunakan 2-3 kali seminggu pada malam hari menggunakan kapas halus.",
        "ingredients": "Salicylic Acid 2%, Tea Tree Extract, Witch Hazel, Willow Bark Extract"
    }
]

added_count = 0
for sp in sample_products:
    existing = db.query(Product).filter(Product.name == sp["name"]).first()
    if not existing:
        p = Product(
            brand=sp["brand"],
            name=sp["name"],
            category=sp["category"],
            description=sp["desc"],
            usage_guidance=sp["usage"],
            ingredients=sp["ingredients"],
            status="active"
        )
        p.set_skin_concern(sp["concerns"])
        db.add(p)
        added_count += 1

db.commit()
print(f"Seed selesai: {added_count} produk baru ditambahkan. Total katalog: {db.query(Product).count()} produk.")
db.close()
