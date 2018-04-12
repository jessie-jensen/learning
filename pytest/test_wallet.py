import pytest
from wallet import Wallet, InsufficientAmount



#
### set fixtures
#

@pytest.fixture()
def empty_wallet():
    '''wallet w/ 0 balance'''
    return Wallet()



@pytest.fixture
def wallet():
    '''wallet w/ 20 balance'''
    return Wallet(20)



#
### test cases
#

def test_default_initial_amount(empty_wallet):
    assert empty_wallet.balance == 0



def test_setting_initial_amount(wallet):
    assert wallet.balance == 20



def test_wallet_add(wallet):
    wallet.add(80)
    assert wallet.balance == 100



def test_wallet_spend(wallet):
    wallet.spend(10)
    assert wallet.balance == 10



def test_wallet_spend_cash_raises_exception_on_insufficient_amount(wallet):
    with pytest.raises(InsufficientAmount):
        wallet.spend(100)



@pytest.mark.parametrize('earned,spent,expected',[
    (30, 10, 40),
    (20, 2, 38),
])
def test_transactions(wallet, earned, spent, expected):
    wallet.add(earned)
    wallet.spend(spent)
    assert wallet.balance == expected
