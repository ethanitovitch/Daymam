from env import BlackJackEnv

from agent import ActorCritic, ActorCriticCombined, DQL

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation


def main():
    plt.style.use('fivethirtyeight')
    env = BlackJackEnv()
    player = DQL()
    total_profit = [0]
    total_shoots = [0]
    fig, ax = plt.subplots()
    for i in range(10000):
        print("STARTING")
        env.reset()
        while env.has_cards_left() and env.user_money > 4:
            print("DEALING CARDS")
            print("USER MONEY:", env.user_money)

            env.new_deal()
            # if env.num_hands_for_group == env.num_hands:
            #     env.new_players()

            pre_states = []
            bets = []
            new_states = []
            actions = []
            final_states = []
            dones = []

            pre_state = env.get_initial_state()
            pre_states.append(pre_state)
            print("Pre state")
            print(pre_state)
            print(" ")

            bet = player.bet(pre_state, env.user_money)
            bets.append(bet)
            print("Bet")
            print(bet)
            print(" ")

            done, new_state = env.place_bet(bet)
            new_states.append(new_state)
            print("New State")
            print(new_state)
            print(" ")

            action = player.act(new_state, env.user_money, env.bet, True)
            actions.append(action)
            print("Action")
            print(action)
            print(" ")

            done, final_state = env.apply(action)
            final_states.append(final_state)
            dones.append(done)
            print("Final State")
            print(final_state)
            print(" ")

            while action == 0 and env.user_total < 21:
                pre_states.append(pre_state)
                bets.append(bet)
                new_states.append(final_state)

                action = player.act(final_state, env.user_money, env.bet, False)
                actions.append(action)
                print("Action")
                print(action)
                print(" ")

                done, final_state = env.apply(action)
                final_states.append(final_state)
                dones.append(done)
                print("Final State")
                print(final_state)
                print(" ")

            reward = env.get_result()
            rewards = [reward for _ in range(len(final_states))]
            print("Reward")
            print(reward)
            print(" ")

            player.remember(pre_states=pre_states, bets=bets, new_states=new_states, actions=actions, rewards=rewards, final_states=final_states, dones=dones)
            print("TRAIN")
            player.replay()
            # player.target_train()

        total_shoots.append(i)
        total_shoots = total_shoots[-10:]

        total_profit.append(env.user_money - 100 + total_profit[i-1])
        total_profit = total_profit[-10:]

        ax.plot(total_shoots, total_profit)

        ax.set(xlabel='Shoot #', ylabel='Profit',
               title='Current Earnings')
        ax.grid()
        plt.pause(0.05)
    plt.show()

if __name__ == "__main__":
    main()
