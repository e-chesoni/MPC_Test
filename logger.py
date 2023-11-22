class Logger:
    def __init__(self):
        self.DEBUG = 1  # 1 = log messages, 0 = don't log
        self.LOG_LEVEL = 3  # 0 = log nothing, 1 = shapes, 2 = variables, 3 = matrices
        self.HEADING = ""

    def set_heading(self, heading):
        self.HEADING = heading

    def log(self, message, arg, log_level, multi_line):
        if self.DEBUG:
            if self.LOG_LEVEL < log_level:
                pass
            else:
                if multi_line:
                    print(f"{self.HEADING} -- {message}: \n {arg}")
                else:
                    print(f"{self.HEADING} -- {message}: {arg}")
