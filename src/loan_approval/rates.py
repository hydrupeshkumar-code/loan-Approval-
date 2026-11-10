"""Dated advertised Indian loan-rate ranges from official lender pages."""

RATE_CHECKED_ON = "10 Nov 2026"

RATE_CATALOG = {
    "Home / mortgage": [
        {
            "lender": "ICICI Bank — standard home loan",
            "minimum": 8.75,
            "maximum": 10.05,
            "source": "https://www.icici.bank.in/personal-banking/loans/home-loan/interest-rates",
            "validity": "Published rates valid through 30 Nov 2026.",
            "details": (
                "The full range spans salaried and self-employed borrowers across loan-size bands. "
                "The page also lists score-specific and pre-approved offers, which are not represented "
                "by this standard-rate range."
            ),
        }
    ],
    "Personal loan": [
        {
            "lender": "ICICI Bank",
            "minimum": 10.25,
            "maximum": 16.50,
            "source": "https://www.icici.bank.in/personal-banking/loans/personal-loan/personal-loan-interest-rates",
            "validity": "Published range checked 10 Nov 2026; verify the lender page for changes.",
            "details": (
                "The page separately reports historic disbursal statistics; those are not the same "
                "as this published range or a personal offer."
            ),
        },
        {
            "lender": "HDFC Bank — salaried rack rate",
            "minimum": 9.99,
            "maximum": 24.00,
            "source": "https://www.hdfc.bank.in/personal-loan/interest-rates-and-charges",
            "validity": "Advertised range checked 10 Nov 2026; verify the lender page for changes.",
            "details": (
                "The page's actual-loan interest statistics cover an earlier quarter and are historical, "
                "not a current quote."
            ),
        },
    ],
}
