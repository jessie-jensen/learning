class InsufficientAmount(Exception):
    pass



class Wallet(object):

    def __init__(self, initial_amount=0):
        self.balance = initial_amount

    def spend(self, amount):
        if self.balance < amount:
            raise InsufficientAmount('Not enough $$$ fool! You need {}'.format(amount))
        self.balance -= amount

    def add(self, amount):
        self.balance += amount