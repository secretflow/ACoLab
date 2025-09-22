def load_txt(instruction_file: str) -> str:
    """
    load .txt file
    Arguments:
        instruction_file: str type, which is the .txt file pth

    Returns:
        instruction: str type, which is the str in the instruction_file

    """
    with open(instruction_file, "r", encoding="utf-8") as f:
        instruction = f.read()
    return instruction
