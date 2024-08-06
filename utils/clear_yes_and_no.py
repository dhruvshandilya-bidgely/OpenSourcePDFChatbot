import os

def clear_yes_no_folder():
    """
    Clearing out YES and NO folder used for debugging before run

    Args:
        docs: (list[langchain_core.documents.base.Document]) Is a list of langchain documents of our pdf.

    Returns:
        None.
    """

    cwd = os.getcwd()

    folder = cwd + '/DEBUG/YES'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    folder = cwd + '/DEBUG/NO'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
