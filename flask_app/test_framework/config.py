"""
Test Framework Configuration
All selectors, URLs, timeouts, and test case definitions.
"""

TARGET_URL = "https://d1u00qm3tt2qg1.cloudfront.net/index.html"

# ── Timeouts (ms) ────────────────────────────────────────────────────────────
TIMEOUT_PAGE_LOAD    = 30_000
TIMEOUT_BOT_RESPONSE = 20_000
TIMEOUT_ELEMENT      = 10_000

# ── CSS Selectors ─────────────────────────────────────────────────────────────
SEL = {
    "email":          "#email",
    "firstName":      "#firstName",
    "lastName":       "#lastName",
    "mobile":         "#mobile",
    "continue_btn":   "button[type='submit']",
    "minimize_card":  ".card-minimize-btn",       # yellow − button on welcome card
    "bot_message":    ".message-bubble-row-bot .message-text",
    "chat_input":     "input[placeholder*='Type here'], .toolbar-input input",
    "all_messages":   ".message-text",
}

# ── Test Suite Definition ─────────────────────────────────────────────────────
# Phases run in order. Within each phase, test cases run sequentially.
#
# action types:
#   wait_for_bot   - just capture the current last bot message (no user input)
#   click_button   - click a button by text content
#   type_message   - type text and press Enter
#
# Each test case defines:
#   name         - display name shown in report
#   action       - one of above
#   value        - button text or message text (None for wait_for_bot)
#   expect       - list of keywords; at least ONE must appear in bot response
#   expect_not   - list of keywords that must NOT appear (optional)
#   critical     - if True, skip remaining tests on failure
#   phase        - logical grouping label

TEST_CASES = [

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 — Initial Boot & Welcome
    # ══════════════════════════════════════════════════════════════
    {
        "phase":    "Phase 1: Boot & Welcome",
        "name":     "TC-01 | Welcome Card Minimized",
        "action":   "check_minimized",
        "value":    None,
        "expect":   [],
        "critical": False,
    },
    {
        "phase":    "Phase 1: Boot & Welcome",
        "name":     "TC-02 | Main Menu Buttons Present",
        "action":   "check_buttons",
        "value":    ["View Products", "Purchase Online",
                     "Schedule Appointment", "View Chat History", "Chat Summary"],
        "expect":   [],
        "critical": False,
    },

    # ══════════════════════════════════════════════════════════════
    # PHASE 2 — Products & Services Navigation
    # ══════════════════════════════════════════════════════════════
    {
        "phase":    "Phase 2: Products & Services",
        "name":     "TC-03 | Click 'View Products & Services'",
        "action":   "click_button",
        "value":    "View Products",
        "expect":   ["Welcome", "help", "How can I help"],
        "critical": False,
    },
    {
        "phase":    "Phase 2: Products & Services",
        "name":     "TC-04 | Products Sub-Menu Contains Voice AI",
        "action":   "check_buttons",
        "value":    ["Voice AI Assistant", "Try Live Demo",
                     "MISRA", "All Products", "Back to Menu"],
        "expect":   [],
        "critical": False,
    },

    # ══════════════════════════════════════════════════════════════
    # PHASE 3 — Voice AI Assistant Info
    # ══════════════════════════════════════════════════════════════
    {
        "phase":    "Phase 3: Voice AI Assistant",
        "name":     "TC-05 | Click 'Voice AI Assistant'",
        "action":   "click_button",
        "value":    "Voice AI Assistant",
        "expect":   ["DigitranVA", "DigiTran", "conversational", "voice",
                     "multimodal", "deploy", "integration"],
        "critical": False,
    },
    {
        "phase":    "Phase 3: Voice AI Assistant",
        "name":     "TC-06 | Response Mentions Key Features",
        "action":   "wait_for_bot",
        "value":    None,
        "expect":   ["responsive", "multimodal", "rich messaging",
                     "live agent", "streaming", "CRM", "calendar"],
        "critical": False,
    },
    {
        "phase":    "Phase 3: Voice AI Assistant",
        "name":     "TC-07 | Response Mentions Use Cases",
        "action":   "wait_for_bot",
        "value":    None,
        "expect":   ["onboarding", "Q&A", "product discovery",
                     "customer service", "voice-enabled", "lead management"],
        "critical": False,
    },
    {
        "phase":    "Phase 3: Voice AI Assistant",
        "name":     "TC-08 | Response Contains Demo Link",
        "action":   "wait_for_bot",
        "value":    None,
        "expect":   ["https://d1u00qm3tt2qg1.cloudfront.net", "Try Live Demo"],
        "critical": False,
    },

    # ══════════════════════════════════════════════════════════════
    # PHASE 4 — Conversational Text Queries
    # ══════════════════════════════════════════════════════════════
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-09 | Query: 'Hello'",
        "action":   "type_message",
        "value":    "Hello",
        "expect":   ["hello", "hi", "help", "Welcome", "assist", "How can"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-10 | Query: 'What can you do?'",
        "action":   "type_message",
        "value":    "What can you do?",
        "expect":   ["service", "Digitran", "question", "provide", "help", "ask"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-11 | Query: 'What is DigiTranVA?'",
        "action":   "type_message",
        "value":    "What is DigiTranVA?",
        "expect":   ["DigiTranVA", "DigiTran", "conversational", "AI",
                     "voice", "assistant", "interface"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-12 | Query: 'What are the key features?'",
        "action":   "type_message",
        "value":    "What are the key features?",
        "expect":   ["feature", "multimodal", "voice", "responsive",
                     "integration", "streaming", "agent"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-13 | Query: 'How does the voice assistant work?'",
        "action":   "type_message",
        "value":    "How does the voice assistant work?",
        "expect":   ["voice", "assistant", "interface", "multimodal",
                     "text", "input", "web"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-14 | Query: 'Can I try a demo?'",
        "action":   "type_message",
        "value":    "Can I try a demo?",
        "expect":   ["demo", "live", "https", "try", "cloudfront", "experience"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-15 | Query: 'How do I integrate this?'",
        "action":   "type_message",
        "value":    "How do I integrate this?",
        "expect":   ["integrat", "embed", "deploy", "frontend",
                     "no-code", "widget", "API", "flexible"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-16 | Query: 'Tell me about MISRA compliance'",
        "action":   "type_message",
        "value":    "Tell me about MISRA compliance",
        "expect":   ["MISRA", "compliance", "C", "safety", "standard",
                     "violation", "analysis"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-17 | Query: 'What industries do you serve?'",
        "action":   "type_message",
        "value":    "What industries do you serve?",
        "expect":   ["industr", "customer", "service", "business",
                     "website", "application", "general"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-18 | Query: 'How accurate is it?'",
        "action":   "type_message",
        "value":    "How accurate is it?",
        "expect":   ["accur", "AI", "voice", "speech", "recognition",
                     "natural", "language", "multimodal"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-19 | Query: 'What is the pricing?'",
        "action":   "type_message",
        "value":    "What is the pricing?",
        "expect":   ["price", "pricing", "cost", "plan", "contact",
                     "purchase", "sales", "demo"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-20 | Query: 'How do I get started?'",
        "action":   "type_message",
        "value":    "How do I get started?",
        "expect":   ["start", "deploy", "embed", "no-code", "launch",
                     "contact", "demo", "integration"],
        "critical": False,
    },
    {
        "phase":    "Phase 4: Conversation",
        "name":     "TC-21 | Query: 'Thank you'",
        "action":   "type_message",
        "value":    "Thank you",
        "expect":   ["thank", "welcome", "help", "pleasure",
                     "assist", "anything", "glad"],
        "critical": False,
    },

    # ══════════════════════════════════════════════════════════════
    # PHASE 5 — Advanced Conversational Queries
    # ══════════════════════════════════════════════════════════════
    {
        "phase":    "Phase 5: Advanced Queries",
        "name":     "TC-22 | Query: 'Do you support multiple languages?'",
        "action":   "type_message",
        "value":    "Do you support multiple languages?",
        "expect":   ["language", "multilingual", "support", "English",
                     "Arabic", "regional", "locale", "translation"],
        "critical": False,
    },
    {
        "phase":    "Phase 5: Advanced Queries",
        "name":     "TC-23 | Query: 'Can you handle voice and text both?'",
        "action":   "type_message",
        "value":    "Can you handle voice and text both?",
        "expect":   ["voice", "text", "multimodal", "input", "both",
                     "speech", "type", "interface"],
        "critical": False,
    },
    {
        "phase":    "Phase 5: Advanced Queries",
        "name":     "TC-24 | Query: 'What makes DigiTranVA different from other chatbots?'",
        "action":   "type_message",
        "value":    "What makes DigiTranVA different from other chatbots?",
        "expect":   ["DigiTran", "unique", "different", "feature",
                     "voice", "multimodal", "integration", "AI"],
        "critical": False,
    },
    {
        "phase":    "Phase 5: Advanced Queries",
        "name":     "TC-25 | Query: 'Is there a free trial available?'",
        "action":   "type_message",
        "value":    "Is there a free trial available?",
        "expect":   ["trial", "demo", "free", "contact", "pricing",
                     "plan", "sales", "try"],
        "critical": False,
    },
    {
        "phase":    "Phase 5: Advanced Queries",
        "name":     "TC-26 | Query: 'How secure is the platform?'",
        "action":   "type_message",
        "value":    "How secure is the platform?",
        "expect":   ["secure", "security", "privacy", "data", "safe",
                     "encrypt", "protection", "confidential"],
        "critical": False,
    },
    {
        "phase":    "Phase 5: Advanced Queries",
        "name":     "TC-27 | Query: 'Can I connect it to my existing CRM?'",
        "action":   "type_message",
        "value":    "Can I connect it to my existing CRM?",
        "expect":   ["CRM", "integrat", "connect", "API", "third-party",
                     "Salesforce", "HubSpot", "system", "flexible"],
        "critical": False,
    },
    {
        "phase":    "Phase 5: Advanced Queries",
        "name":     "TC-28 | Query: 'What is MISRA and why is it important?'",
        "action":   "type_message",
        "value":    "What is MISRA and why is it important?",
        "expect":   ["MISRA", "standard", "C", "C++", "safety",
                     "compliance", "critical", "automotive", "analysis"],
        "critical": False,
    },
    {
        "phase":    "Phase 5: Advanced Queries",
        "name":     "TC-29 | Query: 'How fast does the bot respond?'",
        "action":   "type_message",
        "value":    "How fast does the bot respond?",
        "expect":   ["fast", "response", "real-time", "stream", "instant",
                     "latency", "quick", "speed", "millisecond"],
        "critical": False,
    },
    {
        "phase":    "Phase 5: Advanced Queries",
        "name":     "TC-30 | Query: 'Can the assistant escalate to a human agent?'",
        "action":   "type_message",
        "value":    "Can the assistant escalate to a human agent?",
        "expect":   ["human", "agent", "escalat", "live", "handoff",
                     "transfer", "support", "staff"],
        "critical": False,
    },
]

# ── Reference answers for accuracy scoring ────────────────────────────────────
REFERENCE_ANSWERS = {
    "Hello":
        "Hello, welcome, help, How can I assist",
    "What can you do?":
        "Kindly ask questions related to the services provided by Digitran Solutions",
    "What is DigiTranVA?":
        "DigiTranVA, conversational AI assistant, voice, text, multimodal, deploy, integration",
    "What are the key features?":
        "responsive design, multimodal, voice input, rich messaging, live agent, streaming, CRM, calendar integration",
    "How does the voice assistant work?":
        "web interface, voice input, text input, multimodal, AI, natural language",
    "Can I try a demo?":
        "live demo, cloudfront URL, try, experience, voice interface",
    "How do I integrate this?":
        "no-code launch, frontend embedding, widget, full-page, API, flexible integration",
    "Tell me about MISRA compliance":
        "MISRA C, C++, safety standard, automated analysis, violation report, compliance checker",
    "What industries do you serve?":
        "general purpose, customer service, business, website, application, various industries",
    "How accurate is it?":
        "AI powered, speech recognition, natural language, multimodal, voice input",
    "What is the pricing?":
        "contact sales, pricing plan, demo, purchase, service",
    "How do I get started?":
        "no-code deploy, embed, contact, demo, integration options",
    "Thank you":
        "welcome, pleasure, help, anything else, glad to assist",
    "Do you support multiple languages?":
        "multilingual support, Arabic, regional languages, locale, translation, language",
    "Can you handle voice and text both?":
        "voice input, text input, multimodal, both, speech, interface",
    "What makes DigiTranVA different from other chatbots?":
        "DigiTranVA, unique, multimodal, voice, rich messaging, integration, AI powered",
    "Is there a free trial available?":
        "demo, free trial, contact sales, pricing plan, try",
    "How secure is the platform?":
        "secure, security, data privacy, encryption, protection",
    "Can I connect it to my existing CRM?":
        "CRM integration, API, connect, third-party, flexible, Salesforce, HubSpot",
    "What is MISRA and why is it important?":
        "MISRA C, C++ standard, safety critical, compliance, automotive, violation analysis",
    "How fast does the bot respond?":
        "real-time response, streaming, fast, low latency, instant",
    "Can the assistant escalate to a human agent?":
        "live agent handoff, escalate, human agent, transfer, support staff",
}
