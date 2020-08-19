import random
import numpy as np

INT_TO_ACTION = {
    0: 'stand',
    1: 'hit',
    2: 'double'
}

INT_TO_AMOUNT = {
    0:  5,
    1: 10,
    2: 25,
    3: 50,
}

class BlackJackEnv:
    def __init__(self, decks=8):
        self.decks = decks
        self.shoot = [32,32,32,32,32,32,32,32,32,128]
        self.cards_in_shoot = random.randint(332, 370)
        self.cards_played = 0

        self.players = []

        # for num in range(random.randint(0, 7)):
        #     self.players.append(Player())

        # self.num_hands_for_group = random.randint(1, 5)
        # self.num_hands = 0

        self.user_total = 0
        self.user_money = 100
        self.dealer_total = 0
        self.bet = 0

    def reset(self):
        self.shoot = [32,32,32,32,32,32,32,32,32,128]
        self.cards_in_shoot = random.randint(332, 370)
        self.cards_played = 0

        self.players = []

        # for num in range(random.randint(0, 7)):
        #     self.players.append(Player())
        #
        # self.num_hands_for_group = random.randint(1, 5)
        # self.num_hands = 0

        self.user_total = 0
        self.user_money = 100
        self.dealer_total = 0
        self.bet = 0

    def new_deal(self):
        self.user_total = 0
        self.dealer_total = 0
        for player in self.players:
            player.total = 0

        # self.num_hands += 1

    def new_shoot(self):
        self.shoot = [32,32,32,32,32,32,32,32,32,128]
        self.cards_in_shoot = random.randint(332, 370)
        self.cards_played = 0

    def new_players(self):
        self.players = []

        # for num in range(random.randint(0, 7)):
        #     self.players.append(Player())

    def get_initial_state(self):
        return np.array([item for sublist in [self.shoot, [len(self.players), self.user_money]] for item in sublist])

    def _deal_card(self):
        done = False
        card = random.randint(1, 10)
        while self.shoot[card - 1] == 0:
            card = random.randint(1, 10)

        self.cards_played += 1
        self.shoot[card - 1] -= 1

        if self.cards_played > self.cards_in_shoot:
            done = True

        if card == 1:
            return (11, done)
        else:
            return (card, done)

    def deal_cards(self):
        total, done = self._deal_card()
        self.user_total += total
        for player in self.players:
            total, done = self._deal_card()
            player.total += total

        total, done = self._deal_card()
        self.dealer_total += total

        total, done = self._deal_card()
        self.user_total += total
        for player in self.players:
            total, done = self._deal_card()
            player.total += total

        return done

    def place_bet(self, bet):
        self.bet = INT_TO_AMOUNT[bet]
        if self.bet > self.user_money:
            self.bet = self.user_money
        elif 0 > self.bet:
            self.bet = 0

        self.user_money -= int(self.bet)
        done = self.deal_cards()

        return (done, np.array([item for sublist in
                [self.shoot, [len(self.players), self.user_money, self.dealer_total, self.user_total, self.bet]] for item
                in sublist]))

    def apply(self, action):
        if INT_TO_ACTION[action] == 'hit':
            total, done = self._deal_card()
            self.user_total += total
        elif INT_TO_ACTION[action] == 'double':
            total, done = self._deal_card()
            self.user_total += total
            self.user_money -= self.bet
            self.bet *= 2
        else:
            done = False
        if self.user_total > 21:
            self.user_total = 0

        return (done, np.array([item for sublist in
                [self.shoot, [len(self.players), self.user_money, self.dealer_total, self.user_total, self.bet]] for item
                in sublist]))

    def get_result(self):
        for player in self.players:
            initial = True
            move = player.make_move(self.dealer_total, initial)
            initial=False
            while move != "stand":
                player.total += self._deal_card()
                if move == "double":
                    break
                if player.total > 21:
                    player.total = 0
                    break
                move = player.make_move(self.dealer_total, initial)

        while self.dealer_total <= 17:
            total, done = self._deal_card()
            self.dealer_total += total
            if self.dealer_total > 21:
                dealer_total = 1
                break

        if 22 > self.user_total > self.dealer_total:
            self.user_money += 2*self.bet
            return 2*self.bet
        elif self.user_total == self.dealer_total and self.user_total <= 21:
            self.user_money += self.bet
            return self.bet
        else:
            return 0

    def has_cards_left(self):
        return self.cards_in_shoot > self.cards_played

class Player():
    def __init__(self):
        self.total = 0

    def make_move(self, dealer_total, initial):
        if self.total < 9:
            move = "hit"
        elif self.total == 9:
            if dealer_total < 3 or dealer_total > 6:
                move = "hit"
            else:
                move = "double"
        elif self.total == 10:
            if dealer_total < 10:
                move = "double"
            else:
                move = "hit"
        elif self.total == 11:
            move = "double"
        elif self.total == 12:
            if dealer_total < 4 or dealer_total > 7:
                move = "hit"
            else:
                move = "stand"
        elif 12 < self.total < 17:
            if dealer_total < 6:
                move = "stand"
            else:
                move = "hit"
        else:
            move = "stand"

        if not initial and move == "double":
            move = "hit"
        return move
