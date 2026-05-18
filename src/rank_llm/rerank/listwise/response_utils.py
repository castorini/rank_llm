def clean_ranking_response(
    response: str,
    *,
    use_alpha: bool = False,
    alph_start_idx: int = ord("A") - 1,
) -> str:
    """
    Normalize model ranking output into whitespace-separated numeric ids.

    Examples:
      "[3] > [1] > [2]" -> "3 1 2"
      "A > C > B" (use_alpha=True) -> "1 3 2"
    """
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    fake_numbers_map = str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉①②③④⑤⑥⑦⑧⑨❶❷❸❹❺❻❼❽❾０１２３４５６７８９🄀🄁🄂🄃🄄🄅🄆🄇🄈🄉",
        "0123456789012345678912345678912345678901234567890123456789",
    )
    response = response.translate(fake_numbers_map)

    new_response = ""
    if use_alpha:
        for c in response:
            if not c.isalpha():
                new_response += " "
            else:
                new_response += str(ord(c) - alph_start_idx)
    else:
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c

    return new_response.strip()
