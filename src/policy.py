import random

class Policy:

    @staticmethod
    def write_header_to_csv(file):
        file.write('id,current_year,version,biz_proc,tariff,interest,sum_insured,premium,current_capital,troubled\n')

    def __init__(self, id, sum_insured, tariff):
        self.id = id
        self.sum_insured = sum_insured
        self.tariff = tariff
        self.interest = 0.02 + tariff * 0.01
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
        self.troubled = False


    def adapt_premium(self, premium_change):
        self.premium += premium_change * self.premium

        self.version += 1
        self.biz_proc = "adapt_premium"
        self.troubled = False


    def add_yearly_interest(self, trouble):
        random.seed()

        if (trouble and random.random() < 0.05):
            self.current_capital += self.current_capital * self.interest * random.uniform(5.0, 10.0)
            self.troubled = True
        else:
            self.current_capital += self.current_capital * self.interest
            self.troubled = False

        self.version += 1
        self.biz_proc = "add_yearly_interest"


    def write_to_csv(self, file):
        txt = '%i,%i,%i,%s,%i,%6f,%6f,%.6f,%.6f,%r\n' % \
              (self.id, self.current_year, self.version, self.biz_proc, self.tariff, self.interest, self.sum_insured,
               self.premium, self.current_capital, self.troubled)

        file.write(txt)
