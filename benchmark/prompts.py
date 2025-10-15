"""
Prompts for MuSciClaims benchmark.
Based on the paper's prompt templates for the D (Direct Decision) experiment.
"""


def get_prompt_d(claim: str, caption: str) -> str:
    """
    Get the Direct Decision (D) prompt.
    This is for models to directly output a decision without reasoning.
    """
    prompt = f"""You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning. Your job is to analyze a given image(s) and its provided caption(s) to decide whether it SUPPORT or CONTRADICT or NEUTRAL the provided claim.

CLAIM: {claim}

IMAGE CAPTION(S): {caption}

Guidelines:
1. Evaluate the claim's plausibility based on visual elements within the image(s).
2. Consider the relevance, meaning, and implications of both the depicted content and the caption(s).
3. Analyze the broader context and scope of the image(s) and caption(s) in relation to the claim.

After completing your analysis, output exactly one JSON object with exactly one key: "decision".
- For "decision", output exactly one word — either "SUPPORT" or "CONTRADICT" or "NEUTRAL" (uppercase, no extra text).

Do NOT add markdown formatting, code fences, or any additional text. The output must start with an opening curly brace {{ and end with a closing curly brace }}.

Example output format:
{{"decision": "SUPPORT"}}

Now, please evaluate the image(s) and caption(s) with respect to the claim provided above."""
    
    return prompt


def get_prompt_rd(claim: str, caption: str) -> str:
    """
    Get the Reasoning then Decision (R→D) prompt.
    This is for models to provide reasoning before decision.
    """
    prompt = f"""You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning. Your job is to analyze a given image(s) and its provided caption(s) to decide whether it SUPPORT or CONTRADICT or NEUTRAL the provided claim.

CLAIM: {claim}

IMAGE CAPTION(S): {caption}

Guidelines:
1. Evaluate the claim's plausibility based on visual elements within the image(s).
2. Consider the relevance, meaning, and implications of both the depicted content and the caption(s).
3. Analyze the broader context and scope of the image(s) and caption(s) in relation to the claim.
4. Think step by step to reach your conclusion, but only provide a concise reasoning statement in the output.

After completing your analysis, output exactly one JSON object with exactly two keys in this order: "reasoning" and "decision".
- For "reasoning", provide a brief (one- or two-sentence) explanation of your analysis.
- For "decision", output exactly one word — either "SUPPORT" or "CONTRADICT" or "NEUTRAL" (uppercase, no extra text).

Do NOT add markdown formatting, code fences, or any additional text. The output must start with an opening curly brace {{ and end with a closing curly brace }}.

Example output format:
{{"reasoning": "The caption confirms the rising trend visible in the image, supporting the claim.", "decision": "SUPPORT"}}

Now, please evaluate the image(s) and caption(s) with respect to the claim provided above."""
    
    return prompt


def get_prompt_internvl_d(claim: str, caption: str) -> str:
    """
    Get the Direct Decision prompt formatted for InternVL-style models.
    Simpler format that works better with some models.
    """
    prompt = f"""This is an image from a scientific paper. The following is the caption of the image.

IMAGE CAPTION(S): {caption}

Using this image, analyze whether the following claim is supported, contradicted or neutral according to the image and caption.

CLAIM: {claim}

Reply with one of the following keywords: SUPPORT, CONTRADICT, NEUTRAL. Do not generate any other text or explanation.

Return your answer in following format:
DECISION: <your decision>"""
    
    return prompt


def parse_decision(response: str) -> str:
    """
    Parse the model's response to extract the decision.
    
    Args:
        response: The model's raw output
        
    Returns:
        One of "SUPPORT", "CONTRADICT", "NEUTRAL", or "UNKNOWN"
    """
    response = response.strip().upper()
    
    # Try to parse JSON format
    import json
    import re
    
    # Try to find JSON in the response
    json_match = re.search(r'\{[^}]+\}', response, re.IGNORECASE)
    if json_match:
        try:
            data = json.loads(json_match.group().replace("'", '"'))
            decision = data.get('decision', data.get('DECISION', '')).upper()
            if decision in ["SUPPORT", "CONTRADICT", "NEUTRAL"]:
                return decision
        except:
            pass
    
    # Try to find DECISION: format
    decision_match = re.search(r'DECISION:\s*(SUPPORT|CONTRADICT|NEUTRAL)', response, re.IGNORECASE)
    if decision_match:
        return decision_match.group(1).upper()
    
    # Look for keywords in response
    if "SUPPORT" in response and "CONTRADICT" not in response and "NEUTRAL" not in response:
        return "SUPPORT"
    elif "CONTRADICT" in response and "SUPPORT" not in response and "NEUTRAL" not in response:
        return "CONTRADICT"
    elif "NEUTRAL" in response and "SUPPORT" not in response and "CONTRADICT" not in response:
        return "NEUTRAL"
    
    # Check for the decision being the last word or standalone
    words = response.split()
    for word in reversed(words):
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in ["SUPPORT", "CONTRADICT", "NEUTRAL"]:
            return clean_word
    
    return "UNKNOWN"


# Map label strings to standardized format
LABEL_MAP = {
    "support": "SUPPORT",
    "contradict": "CONTRADICT",
    "neutral": "NEUTRAL",
    "SUPPORT": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "NEUTRAL": "NEUTRAL",
}


def normalize_label(label: str) -> str:
    """Normalize a label to uppercase standard format."""
    return LABEL_MAP.get(label, LABEL_MAP.get(label.lower(), "UNKNOWN"))


def get_gallop_prompt(claim: str, caption: str, drop_caption: bool = False, drop_claim: bool = False) -> str:
    """
    Get the GalLoP prompt with optional dropout.
    Global Component: Caption (dropped if drop_caption=True)
    Local Component: Claim (dropped if drop_claim=True)
    """
    
    # Construct components (GalLoP strategy: Global=Caption, Local=Claim)
    global_prompt = f"IMAGE CAPTION(S): {caption}" if not drop_caption else "IMAGE CAPTION(S): [Unavailable]"
    local_prompt = f"CLAIM: {claim}" if not drop_claim else "CLAIM: [Unavailable]"
    
    prompt = f"""You are an AI model tasked with verifying claims related to visual evidence.

{local_prompt}

{global_prompt}

Guidelines:
1. Analyze the image and the provided text.
2. Determine if the image SUPPORT, CONTRADICT, or is NEUTRAL to the claim.

Output exactly one JSON object with key "decision".
Value must be one of: "SUPPORT", "CONTRADICT", "NEUTRAL".

Example:
{{"decision": "SUPPORT"}}
"""
    return prompt
