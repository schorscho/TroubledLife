import random

class Policy:

    @staticmethod
    def write_header_to_csv(file):
        file.write('id,current_year,version,biz_proc,premium,current_capital,troubled\n')

    def __init__(self, id, sum_insured):
        self.id = id
        self.interest = 0.04
        self.premium = sum_insured / 100
        self.current_capital = 0.0
        self.current_year = 0
        self.version = 0
        self.troubled = False
        self.biz_proc = "initial"

    def collect_premium(self, year):
        self.current_capital += self.premium

        self.current_year = year
        self.version = 1
        self.biz_proc = "collect_premium"

    def adapt_premium(self, premium_change):
        self.premium += premium_change * self.premium

        self.version += 1
        self.biz_proc = "adapt_premium"

    def add_yearly_interest(self):
        random.seed()

        if (random.random() < 0.05):
            self.current_capital += self.current_capital * self.interest * random.uniform(5.0, 10.0)
            self.troubled = True
        else:
            self.current_capital += self.current_capital * self.interest

        self.version += 1
        self.biz_proc = "add_yearly_interest"

    def write_to_csv(self, file):
        txt = '%i,%i,%i,%s,%.6f,%.6f,%r\n' % (self.id, self.current_year, self.version, self.biz_proc, self.premium, self.current_capital, self.troubled)

        file.write(txt)
