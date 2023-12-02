from logger_config import logger


def load_file(filename):
    try:
        with open(filename, "r") as file:
            lines = [line.rstrip("\n") for line in file]
            logger.info(f"Successfully read file {filename}")
            return lines
    except FileNotFoundError:
        logger.error(f"File {filename} not found")
        return []
    except Exception as e:
        logger.error(f"Error occurred while reading file {filename}: {str(e)}")
        return []
