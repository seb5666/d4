import random

class BlackJack:
    
    def __init__(self):
        #list of card values
        self.card_names = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King"]
        self.card_values = [1,2,3,4,5,6,7,8,9,10,10,10,10]

        #cards held by players: cards[i] = x indicates that the person has x times the ith card
        self.player_cards = [0 for i in range(len(self.card_names))]
        self.dealer_cards = [0 for i in range(len(self.card_names))]
        
        #intitialize player hand
        [c1, c2] = self.draw_cards(n = 2)
        self.player_cards[c1] += 1
        self.player_cards[c2] += 1
        while(self.sum_cards(self.player_cards) < 12):
            [card] = self.draw_cards(n=1)
            self.player_cards[card] += 1

        #intitialize dealer hand
        [c1, c2] = self.draw_cards(n = 2)
        self.dealer_cards[c1] += 1
        self.dealer_cards[c2] += 1
        self.dealer_showing_card = c1

        #initialize episode
        self.player_stuck = False
        self.player_bust = False
        self.game_ended = False
        self.game_outcome = 0
        
        #test for immediate win or draw
        if (self.sum_cards(self.player_cards) == 21):
            dealer_sum = self.sum_cards(self.dealer_cards)
            if dealer_sum == 21:
                self.game_outcome = 0
            else:
                self.game_outcome = 1
            self.game_ended = True

        #self.print_player_hand()
        #print()
        #self.print_dealer_hand()

    
    def print_player_hand(self):
        print("Player cards:")
        print(self.print_hand(self.player_cards))
        print("Total: {}".format(self.sum_cards(self.player_cards)))
        print("Has usable ace: {}".format(self.has_usable_ace(self.player_cards)))
        
    def print_dealer_hand(self):
        print("Dealer shown card: {}".format(self.card_names[self.dealer_showing_card]))
        print("Dealer cards:")
        print(self.print_hand(self.dealer_cards))
        print("Total: {}".format(self.sum_cards(self.dealer_cards)))

    def draw_cards(self, n = 1):
        return random.choices(range(len(self.card_names)), k=n)

    def print_hand(self, cards):
        res = ""
        for i,x in enumerate(cards):
            if x > 0:
                res += "{} x: {}\n".format(x, self.card_names[i])
        return res[:-1]
   
    def has_usable_ace(self, cards):
        total = 0
        for i, x in enumerate(cards):
            total += x * self.card_values[i]
        num_aces = cards[0]
        return (num_aces >= 1 and total <= 11)

    def sum_cards(self, cards):
        total = 0
        for i, x in enumerate(cards):
            total += x * self.card_values[i]
        num_aces = cards[0]
        while(total <= 11 and num_aces > 0):
            total += 10
            num_aces -= 1
        return total

    def take_action(self, a):
        if a == "hit":
            self.hit(self.player_cards)
        else:
            self.player_stuck = True

        if (self.player_stuck or self.player_bust):
            while(self.sum_cards(self.dealer_cards) < 17):
                self.hit(self.dealer_cards)

            player_total = self.sum_cards(self.player_cards)
            dealer_total = self.sum_cards(self.dealer_cards)

            if (player_total <= 21 and (dealer_total < player_total or dealer_total > 21)):
                self.game_outcome = 1
            elif (player_total > 21 and dealer_total > 21):
                self.game_outcome = 0
            elif (player_total == dealer_total):
                self.game_outcome = 0
            else:
                self.game_outcome = -1

            self.game_ended = True

        return (self.get_state(), 0 if not(self.game_ended) else self.game_outcome)

    def hit(self, cards):
        [card] = self.draw_cards(n=1)
        cards[card] += 1

        if (self.sum_cards(self.player_cards) > 21):
            self.player_bust = True

    # State consists of:
    # - Player's current sum (12 - 21)
    # - Dealer's showing card (Ace- 10)
    # - Player holding a usable Ace, i.e. if he has an ace that could count as 1 (True/False)
    def get_state(self):
        player_sum = self.sum_cards(self.player_cards)
        dealer_card = self.dealer_showing_card
        usable_ace = self.has_usable_ace(self.player_cards)
        return (player_sum, dealer_card, usable_ace)
    
    def print_state(self):
        (ps, dc, ua) = self.get_state()
        print(ps, self.card_names[dc], ua)

if __name__ == "__main__":
    Game = BlackJack()
    while (not(Game.game_ended)):
        Game.print_state()
        action = input()
        while (action != 'h' and action != 's'):
            print("Wrong action")
            action = input()
        Game.take_action("hit" if action == 'h' else "stick")

    Game.print_player_hand()
    Game.print_dealer_hand()
    print("Outcome: {}".format(Game.game_outcome))


