import sys
from src.logger import logging
def custom_error(error,error_detail:sys) :
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_message = "The error occurs in {} python script and the line number is {} and the error is {}".format(file_name,line_no,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = custom_error(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
if __name__ == '__main__' :
    try :
        a=1/0
        print("Hello World")
    except Exception as e :
        error_message = custom_error(e,error_detail=sys)
        logging.info(error_message)
        raise Exception(error_message)

