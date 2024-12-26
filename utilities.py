def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to timestamp string in MM:SS format.
    
    Args:
        seconds (float): Number of seconds
        
    Returns:
        str: Timestamp in MM:SS format
    """

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"