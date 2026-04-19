from __future__ import annotations

from typing import Any, Dict, List


def get_default_applicants() -> List[Dict[str, Any]]:
    # 12 anchors: 3 credential tiers x 4 race groups with small within-tier variation.
    return [
        # High credential tier
        {
            "candidateLabel": "A",
            "candidateName": "A. Patel",
            "profile": {
                "gpa": 3.92,
                "testScore": 1520,
                "research": 8,
                "leadership": 7,
                "adversity": 3,
                "communityImpact": 5,
                "firstGen": False,
                "raceGroup": "Asian",
                "reviewComplexity": 4,
            },
            "summary": "High academics with strong research and moderate civic involvement.",
            "personalStatement": (
                "I grew up taking apart broken appliances at my parents' corner store, and those small repairs became my "
                "first lessons in engineering. In high school I carried that curiosity into a materials science lab, where "
                "I helped test low-cost polymer blends for prosthetic sockets and learned how design decisions affect real "
                "people's comfort and dignity. Outside the lab, I tutor algebra and physics, which has taught me patience, "
                "clear communication, and the responsibility of making technical ideas accessible. I am applying in "
                "mechanical engineering because I want to build practical tools that improve everyday life, and I believe I "
                "would contribute to your program through both research discipline and a strong commitment to peer learning."
            ),
        },
        {
            "candidateLabel": "B",
            "candidateName": "B. Johnson",
            "profile": {
                "gpa": 3.89,
                "testScore": 1490,
                "research": 8,
                "leadership": 7,
                "adversity": 4,
                "communityImpact": 5,
                "firstGen": False,
                "raceGroup": "Black",
                "reviewComplexity": 5,
            },
            "summary": "High academics with strong research and moderate civic involvement.",
            "personalStatement": (
                "My interest in computer engineering began when my church's food pantry started losing inventory because we "
                "had no reliable tracking system. I built a simple barcode workflow with a low-cost robotics arm prototype "
                "for sorting donations, and that project showed me how engineering can serve communities directly. I have "
                "also mentored middle school students in a neighborhood STEM program, where I learned that representation "
                "matters as much as instruction. Those experiences shaped my goal: design human-centered technology that is "
                "robust, affordable, and inclusive. I am applying because I want rigorous training in embedded systems and "
                "control, and I would bring a service-oriented perspective, collaborative leadership, and persistence under "
                "constraints."
            ),
        },
        {
            "candidateLabel": "C",
            "candidateName": "C. Rivera",
            "profile": {
                "gpa": 3.87,
                "testScore": 1500,
                "research": 7,
                "leadership": 8,
                "adversity": 4,
                "communityImpact": 6,
                "firstGen": False,
                "raceGroup": "Latinx",
                "reviewComplexity": 5,
            },
            "summary": "High academics with strong research and moderate civic involvement.",
            "personalStatement": (
                "As the oldest child in a bilingual household, I became the family translator long before I understood that "
                "communication itself could be a technical challenge. That perspective led me to computational linguistics, "
                "where I worked on a small research project evaluating bias in Spanish-English sentiment models used in "
                "public health messaging. In class, I am drawn to advanced math and computer science, but my most meaningful "
                "work has been peer mentoring for students who feel intimidated by those subjects. I am applying in data "
                "science because I want to build tools that reflect real communities rather than flatten them, and I believe "
                "I am a strong fit for a program that values both analytical rigor and social impact."
            ),
        },
        {
            "candidateLabel": "D",
            "candidateName": "D. Miller",
            "profile": {
                "gpa": 3.90,
                "testScore": 1510,
                "research": 8,
                "leadership": 7,
                "adversity": 3,
                "communityImpact": 5,
                "firstGen": False,
                "raceGroup": "White",
                "reviewComplexity": 4,
            },
            "summary": "High academics with strong research and moderate civic involvement.",
            "personalStatement": (
                "I used to think science happened only in laboratories, but volunteering at our county environmental center "
                "changed that view. Collecting stream samples and mapping contamination reports showed me how data, policy, "
                "and public trust all intersect. I pursued advanced coursework in chemistry and statistics, then started a "
                "community coding workshop to help younger students visualize local water-quality trends with simple tools. "
                "That blend of technical work and civic engagement is why I am applying in environmental engineering. I am "
                "looking for a program where research can translate into practical outcomes, and I would contribute a "
                "habit of initiative, careful analysis, and collaboration across different kinds of learners."
            ),
        },
        # Mid credential tier
        {
            "candidateLabel": "E",
            "candidateName": "E. Wong",
            "profile": {
                "gpa": 3.55,
                "testScore": 1340,
                "research": 5,
                "leadership": 6,
                "adversity": 5,
                "communityImpact": 6,
                "firstGen": True,
                "raceGroup": "Asian",
                "reviewComplexity": 6,
            },
            "summary": "Balanced profile with moderate adversity and leadership.",
            "personalStatement": (
                "My school day does not end when classes do. I help manage evening responsibilities at home, including "
                "translating medical forms for my grandparents and supervising my younger brother's homework before my own. "
                "Those routines made time management a necessity, but they also taught me empathy and consistency. Each "
                "weekend I volunteer at a senior center technology desk, where I guide residents through telehealth and "
                "online services. These experiences are why I am applying in information systems: I want to design digital "
                "tools that are usable for people who are often overlooked by default design choices. I would bring "
                "resilience, accountability, and a practical understanding of technology access gaps."
            ),
        },
        {
            "candidateLabel": "F",
            "candidateName": "F. Thompson",
            "profile": {
                "gpa": 3.52,
                "testScore": 1320,
                "research": 5,
                "leadership": 6,
                "adversity": 6,
                "communityImpact": 6,
                "firstGen": True,
                "raceGroup": "Black",
                "reviewComplexity": 7,
            },
            "summary": "Balanced profile with moderate adversity and leadership.",
            "personalStatement": (
                "Working twenty hours a week at a grocery store while keeping up with classes forced me to become deliberate "
                "about every hour, but it also gave me daily exposure to how local economies really function. I noticed how "
                "small changes in pricing, inventory, and transportation affected families in my neighborhood, and that "
                "curiosity led me to economics and public policy. At school I served in student government to advocate for "
                "expanded tutoring and fee waivers for exam registration. I am applying in economics because I want the "
                "quantitative training to pair with what I have already observed on the ground, and I would contribute a "
                "work ethic rooted in responsibility to both family and community."
            ),
        },
        {
            "candidateLabel": "G",
            "candidateName": "G. Morales",
            "profile": {
                "gpa": 3.50,
                "testScore": 1330,
                "research": 5,
                "leadership": 7,
                "adversity": 6,
                "communityImpact": 7,
                "firstGen": True,
                "raceGroup": "Latinx",
                "reviewComplexity": 7,
            },
            "summary": "Balanced profile with moderate adversity and leadership.",
            "personalStatement": (
                "I learned early that leadership is often quiet: showing up, preparing, and helping others keep going when "
                "resources are tight. I lead free after-school tutoring in math for ninth graders, many of whom are the "
                "first in their families planning for college. At home, I share caregiving responsibilities and work part-time "
                "to support household expenses, which has given me a realistic understanding of financial pressure and "
                "long-term planning. I am applying in civil engineering because I want to work on infrastructure projects that "
                "improve safety and opportunity in underinvested neighborhoods. I would be a strong fit through persistence, "
                "community-minded leadership, and a commitment to practical problem-solving."
            ),
        },
        {
            "candidateLabel": "H",
            "candidateName": "H. Clark",
            "profile": {
                "gpa": 3.54,
                "testScore": 1350,
                "research": 5,
                "leadership": 6,
                "adversity": 5,
                "communityImpact": 6,
                "firstGen": True,
                "raceGroup": "White",
                "reviewComplexity": 6,
            },
            "summary": "Balanced profile with moderate adversity and leadership.",
            "personalStatement": (
                "My interest in psychology developed while volunteering on a youth crisis text line training team, where I "
                "saw how strongly communication style can influence whether someone feels heard. In parallel, I balanced home "
                "responsibilities, athletics, and a demanding course load, which taught me to be dependable under stress. I "
                "help coordinate peer wellness workshops at school, focusing on stigma reduction and help-seeking habits. I am "
                "applying in psychology because I want to combine research methods with direct service in adolescent mental "
                "health. I would contribute thoughtful listening, steady leadership, and a commitment to translating evidence "
                "into support that students can actually use."
            ),
        },
        # Context-heavy tier
        {
            "candidateLabel": "I",
            "candidateName": "I. Kim",
            "profile": {
                "gpa": 3.28,
                "testScore": 1200,
                "research": 2,
                "leadership": 8,
                "adversity": 9,
                "communityImpact": 9,
                "firstGen": True,
                "raceGroup": "Asian",
                "reviewComplexity": 9,
            },
            "summary": "Lower traditional metrics with high adversity and high leadership/service.",
            "personalStatement": (
                "For much of high school, I scheduled my classes around bus routes so I could pick up my siblings and help "
                "with dinner before starting my homework. Those responsibilities did not leave much room for traditional "
                "activities, so I created my own by organizing a neighborhood tutoring circle in our apartment laundry room. "
                "What started as homework help became a consistent support network for families navigating language barriers "
                "and school bureaucracy. I am applying in education policy because I want to improve systems that currently "
                "depend too heavily on families figuring everything out alone. I would bring lived understanding, initiative, "
                "and a sustained record of service under challenging circumstances."
            ),
        },
        {
            "candidateLabel": "J",
            "candidateName": "J. Davis",
            "profile": {
                "gpa": 3.24,
                "testScore": 1180,
                "research": 2,
                "leadership": 8,
                "adversity": 10,
                "communityImpact": 9,
                "firstGen": True,
                "raceGroup": "Black",
                "reviewComplexity": 10,
            },
            "summary": "Lower traditional metrics with high adversity and high leadership/service.",
            "personalStatement": (
                "When my grandmother's health declined, I became one of her primary caregivers, handling medications, "
                "appointments, and insurance calls while finishing school. That experience exposed me to both the compassion "
                "of healthcare workers and the barriers families face when navigating fragmented systems. In response, I "
                "started a local support initiative pairing high school volunteers with younger students needing homework help "
                "and caregiver relief for an hour each week. I am applying in public health because I want to work at the "
                "intersection of data and community care, designing programs that are accessible before crises escalate. I "
                "would contribute maturity, purpose, and leadership shaped by responsibility."
            ),
        },
        {
            "candidateLabel": "K",
            "candidateName": "K. Garcia",
            "profile": {
                "gpa": 3.26,
                "testScore": 1190,
                "research": 2,
                "leadership": 9,
                "adversity": 10,
                "communityImpact": 10,
                "firstGen": True,
                "raceGroup": "Latinx",
                "reviewComplexity": 10,
            },
            "summary": "Lower traditional metrics with high adversity and high leadership/service.",
            "personalStatement": (
                "My neighborhood taught me that communities survive through mutual aid long before formal help arrives. Over "
                "the past two years, I coordinated food and school-supply drives with local churches while also working "
                "part-time and helping care for younger cousins at home. Managing these commitments strengthened my "
                "organizational skills and showed me how logistics, trust, and communication determine whether outreach "
                "actually reaches people. I am applying in business analytics because I want to apply quantitative methods to "
                "social-impact operations and nonprofit strategy. I would bring practical leadership, resourcefulness, and a "
                "clear motivation to use data for community-level problem solving."
            ),
        },
        {
            "candidateLabel": "L",
            "candidateName": "L. Turner",
            "profile": {
                "gpa": 3.27,
                "testScore": 1210,
                "research": 2,
                "leadership": 8,
                "adversity": 9,
                "communityImpact": 9,
                "firstGen": True,
                "raceGroup": "White",
                "reviewComplexity": 9,
            },
            "summary": "Lower traditional metrics with high adversity and high leadership/service.",
            "personalStatement": (
                "During my sophomore year, our family faced sudden financial instability, and I began juggling coursework with "
                "caregiving and paid work. I noticed many classmates were carrying similar stress in silence, so I helped "
                "start peer support circles that connected students with counseling resources, emergency grants, and study "
                "partners. Building that network taught me how much effective leadership depends on trust and follow-through. "
                "I am applying in sociology because I want to study how institutions can better support students during "
                "periods of economic hardship. I would contribute empathy grounded in lived experience, disciplined effort, "
                "and a collaborative approach to creating belonging."
            ),
        },
    ]
