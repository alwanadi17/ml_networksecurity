import sys
import traceback

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        self.formatted_traceback = traceback.format_stack()

    def __str__(self):
        divider = "=" * 40
        return (
            f"\n{divider}\n"
            f"❌ [ERROR {self.code}]: {self.error_message}\n"
            f"{divider}\n"
            f"📜 System Info: {super().__str__()}\n"
            f"📍 Location Trace:\n{self.formatted_traceback}"
            f"{divider}"
        )