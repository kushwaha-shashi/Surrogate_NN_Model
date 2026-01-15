PROJECT_DIR = "/home/docker_user"
OUTPUT_SUBDIR = PROJECT_DIR + "/output"
INPUT_SUBDIR = PROJECT_DIR + "/input"


def get_output_path(output_subdir):
    return OUTPUT_SUBDIR + "/" + output_subdir


def get_input_path(input_subdir):
    return INPUT_SUBDIR + "/" + input_subdir