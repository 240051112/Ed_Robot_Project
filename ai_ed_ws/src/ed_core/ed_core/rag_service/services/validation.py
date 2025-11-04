def validate_query(question: str) -> bool:
    if not question or len(question.strip()) < 5:
        return False
    # guard against empty control chars only
    return any(ch.isalnum() for ch in question)

def validate_command(command: str, parameters: dict) -> bool:
    if not command or not isinstance(command, str):
        return False
    # Add specific command validation as needed (e.g., gripper open/close)
    return True
