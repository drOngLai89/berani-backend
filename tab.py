"""
Tribunal for Anti-bully (TAB) facts and links, taken only from the Tribunal's own pages.

Berani's AI may state Tribunal facts ONLY from TAB_FACTS. Anything else (whether a case
qualifies, deadlines, outcomes, legal advice, complaint status) goes to the Tribunal itself
through Tanya Kami. Update this file when tab.bheuu.gov.my changes; no app release needed.

Checked against tab.bheuu.gov.my on 24 Oct 2026: /halaman/cara-mengadu, /soalan-lazim,
/hubungi, /tanya-kami, /aduan/mula, /muat-turun?category=borang, /lampiran-a/download.
"""
from __future__ import annotations

import os

SITE = "https://tab.bheuu.gov.my"
LINKS = {
    "portal": SITE,
    "startComplaint": f"{SITE}/aduan/mula",
    "checkStatus": f"{SITE}/aduan/status",
    "askUs": f"{SITE}/tanya-kami",
    "howTo": f"{SITE}/halaman/cara-mengadu",
    "faq": f"{SITE}/soalan-lazim",
    "typesOfBullying": f"{SITE}/jenis-buli",
    "victimRights": f"{SITE}/hak-mangsa",
    "borang1": f"{SITE}/storage/resources/4a20f4c0-5064-497f-aa7a-92df952fc4ae.pdf",
    "lampiranA": f"{SITE}/lampiran-a/download",
}
EMAIL = "tab@bheuu.gov.my"
HELPDESK = "helpdesk.tribunalantibuli@bheuu.gov.my"
PHONE = "03-2271 1000"
HOURS = "Monday to Friday, 9am to 4.30pm"

TAB_FACTS = [
    "Who can complain: the victim, or the mother, father or guardian of a child victim. Teachers cannot file a complaint.",
    "Bullying under the supervision of a school or other institution goes first to that institution's anti-bully committee (JPPKB). "
    "The Tribunal can then hear it if no action was taken, the complaint was not handled properly, or the committee refers it with the complainant's consent.",
    "A child who was bullied outside any school or institution's supervision can complain to the Tribunal directly.",
    "The Tribunal cannot hear a case that has already been started in a civil court.",
    "Complaints are made online through e-TAB on tab.bheuu.gov.my ('Mulakan Aduan'), which starts with a short screening. "
    "Prepare evidence (messages, screenshots, photos, recordings, witness details), a police report number if there is one, "
    "details of the victim, the person who bullied and witnesses if known, and Lampiran A if it happened under a school or institution's supervision.",
    "Lampiran A is a confirmation from the school or institution's anti-bully committee (JPPKB) about what action it took, signed by the principal or an authorised officer with the official stamp. "
    "The Tribunal provides it as a download with a cover letter.",
    "The complaint form is Borang 1 (Aduan Buli). The other side replies with Borang 2 within 14 working days of receiving it.",
    "After a complaint, the Tribunal first reviews it. If everyone agrees, it can be settled early, and the Tribunal may refer it to the Legal Aid Department for mediation. "
    "Otherwise there is a private hearing, which may be held online.",
    "Lawyers are allowed only when the Tribunal finds the case involves complex legal issues. The Tribunal may refer a child without a lawyer for legal aid.",
    "Making a complaint does not cost anything.",
    f"Check a complaint's status at {LINKS['checkStatus']}. Questions: Tanya Kami at {LINKS['askUs']} (the Tribunal replies within 3 working days), "
    f"{EMAIL}, or {PHONE} ({HOURS}). Login or form problems: {HELPDESK}.",
]


def facts_block() -> str:
    return "\n".join(f"- {f}" for f in TAB_FACTS)


def config() -> dict:
    """Read by the app at start. Switches here change the app without a store release."""
    return {
        "tab": {
            "links": LINKS,
            "email": EMAIL,
            "helpdesk": HELPDESK,
            "phone": PHONE,
            "hours": HOURS,
        },
        # Turned on only after the Tribunal confirms that an emailed Borang 1 is accepted as a filing.
        "emailFiling": os.getenv("TAB_EMAIL_FILING", "").lower() in ("1", "true", "yes"),
        # Turned on only if the Tribunal launches an official WhatsApp channel.
        "tabWhatsApp": os.getenv("TAB_WHATSAPP", "") or None,
        "voice": True,
    }
