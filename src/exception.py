import sys

def get_error_detail(error_msg, error_detail:sys):
    # retrieveing the traceback object using sys_obj.exc_info()
    _, _, exc_tb = error_detail.exc_info()
    error_file = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(error_file,exc_tb.tb_lineno,str(error_msg))


    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message=get_error_detail(error_message,error_detail=error_detail)


    def __str__(self):
        return self.error_message