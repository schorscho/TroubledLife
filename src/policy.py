import random

class Policy:

    @staticmethod
    def write_header_to_csv(file):
        file.write('id,current_year,version,premium,current_capital,troubled\n')

    def __init__(self, id, sum_insured):
        self.id = id
        self.interest = 0.04
        self.premium = sum_insured / 100
        self.current_capital = 0.0
        self.current_year = 0
        self.version = 0
        self.troubled = False

    def collect_premium(self, year):
        self.current_capital += self.premium

        self.current_year = year
        self.version = 1

    def adapt_premium(self, premium_change):
        self.premium += premium_change * self.premium

        self.version += 1

    def add_yearly_interest(self):
        if (random.random() < 0.05):
            self.current_capital += self.current_capital * self.interest * random.uniform(5.0, 10.0)
            self.troubled = True
        else:
            self.current_capital += self.current_capital * self.interest

        self.version += 1

    def write_to_csv(self, file):
        txt = '%i,%i,%i,%.6f,%.6f,%r\n' % (self.id, self.current_year, self.version, self.premium, self.current_capital, self.troubled)

        file.write(txt)
