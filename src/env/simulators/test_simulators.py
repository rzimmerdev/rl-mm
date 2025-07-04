import unittest
import matplotlib.pyplot as plt
import numpy as np

from env.lob import Order
from .market_simulator import MarketSimulator


class TestMarketSimulator(unittest.TestCase):
    def test_normality(self):
        num_samples = 1000
        market_simulator = MarketSimulator(
            100,
            .02,
            .0,
            .01,
            0,
            0,
            1,
            event_size_mean=num_samples,
        )
        # test if orders generated are normally distributed
        midprice = 100
        rf = 0.02
        orders = market_simulator._sample_orders(midprice)

        prices = np.array([order.price for order in orders])

        plt.hist(prices, bins=50)
        plt.axvline(np.mean(prices), color='black', linestyle='--')
        plt.axvline(100 * np.exp(rf), color='black')
        plt.title("Price distribution")
        plt.show()

    def test_manual_sample(self):
        trading_hours = 6.5
        sim = MarketSimulator(
            435,
            .15,
            .0,
            .01 * np.sqrt(60),
            1e-1,
            1e-2,
            1 / 252 / trading_hours / 60,
            event_size_mean=.1,
        )

        p = sim.starting_value
        ps = [p]
        pasks = []
        tasks = []
        pbids = []
        tbids = []
        ts = [0]
        t = 0

        while t < 1 / 252:
            ask_size = sim.event_size_distribution(sim.event_size_mean)
            bid_size = sim.event_size_distribution(sim.event_size_mean)

            ask_price = sim.ask_process.sample(p, 1 / 252 / 6.5 / 60, ask_size)
            bid_price = sim.bid_process.sample(p, 1 / 252 / 6.5 / 60, bid_size)

            asks = [Order(side='ask', price=price, quantity=100) for price in ask_price]
            bids = [Order(side='bid', price=price, quantity=100) for price in bid_price]
            orders = np.array(asks + bids)
            np.random.shuffle(orders)

            for order in orders:
                sim.lob.send_order(order)
            p = sim.midprice()
            ps.append(p)
            ts.append(t)
            if sim.lob.asks.bottom() is not None:
                pasks.append(sim.lob.asks.bottom().value.price)
                tasks.append(t)
            elif len(pasks) > 0:
                pasks.append(pasks[-1])
                tasks.append(t)
            if sim.lob.bids.top() is not None:
                pbids.append(sim.lob.bids.top().value.price)
                tbids.append(t)
            elif len(pbids) > 0:
                pbids.append(pbids[-1])
                tbids.append(t)
            t += sim.dt

        ps = np.array(ps)
        returns = np.diff(ps) / ps[:-1]
        vol = np.std(returns) * np.sqrt(252 * trading_hours * 60)
        vol = vol * 100
        print(f"Annualized volatility: {vol:.2f}%")  # => 2% to 3%

        # annualized returns
        mean_returns = np.mean(returns)
        annualized_returns = mean_returns * 252 * trading_hours * 60
        annualized_returns = annualized_returns * 100
        print(f"Annualized returns: {annualized_returns:.2f}%")

        # plot midprice, ask, bids
        plt.plot(ts, ps)
        plt.plot(tasks, pasks, 'r')
        plt.plot(tbids, pbids, 'g')
        plt.title("Midprice")
        plt.xlabel("Time (hours)")
        plt.ylabel("Price")
        plt.show()

        w = 10
        ps = [np.mean(ps[i:i + w]) for i in range(len(ps) - w)]
        ts = ts[:-w]
        ts = [t * 6.5 * 252 for t in ts]

        # plot ts only 10 minute per ten minute (so if ts = [0, 5, 6, 11, 14, 19, 25], take only [0, 11, 19])
        downsampled_ts = []
        downsampled_ps = []

        # Start by adding the first point
        downsampled_ts.append(ts[0])
        downsampled_ps.append(ps[0])

        target_interval = 10 / 60

        # Iterate through the data and select points where the time difference is roughly 10 minutes
        last_time = ts[0]
        for i in range(1, len(ts)):
            if ts[i] - last_time >= target_interval:  # Check if the gap is close to the target
                downsampled_ts.append(ts[i])
                downsampled_ps.append(ps[i])
                last_time = ts[i]

        ps = downsampled_ps
        ts = downsampled_ts

        # dimension = 2 x1, axis = 425 to 440
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ts, ps)
        ax.set_title("Midprice")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Price")
        ax.set_xlim(0, 6.5)
        plt.show()

    def test_step(self):
        trading_hours = 6.5
        sim = MarketSimulator(
            415,
            .15,
            .0,
            .15 * np.sqrt(60),
            1,
            1e-2,
            1 / 252 / trading_hours / 60,
            event_size_mean=1,
        )

        p = []
        t = []
        while sim.market_timestep < 1 / 252:
            transactions = sim.step()
            p.append(sim.midprice())
            t.append(sim.market_timestep)

        # get only every 10 minutes

        # Assuming t and p are populated
        t_scaled = np.linspace(0, 390, len(t))  # Map t to 0-390 minutes
        t_resampled = t_scaled[::10]  # Get only every 10 minutes
        p_resampled = p[::10]

        # Convert t_resampled to time labels (HH:MM)
        def format_time(minutes):
            total_minutes = 9 * 60 + 30 + minutes  # Start from 9:30
            hours = total_minutes // 60
            mins = total_minutes % 60
            return f"{int(hours):02}:{int(mins):02}"

        # Generate time labels for the resampled t values
        time_labels = [format_time(m) for m in t_resampled]

        # Plot with corrected labels
        plt.figure(figsize=(10, 6))
        plt.plot(t_resampled, p_resampled, label="Midprice")
        plt.xticks(t_resampled[::3], time_labels[::3], rotation=45)  # Label every 30 minutes
        plt.ylim(410, 420)
        plt.xlabel("Time (HH:MM)")
        plt.ylabel("Midprice")
        plt.title("Midprice from 9:30 to 16:00")
        plt.legend()
        plt.grid()
        plt.show()

    def test_set_order(self):
        sim = MarketSimulator(
            435,
            .15,
            .0,
            .15 * np.sqrt(60),
            1e-3,
            1e-2,
            1 / 252 / 6.5 / 60,
            event_size_mean=1,
        )

        sim.set_order(bid=Order(side='bid', price=100, quantity=100))

        print(sim.lob.orders)

        sim.set_order(bid=Order(side='bid', price=100 + sim.order_eps, quantity=100))

        print(sim.lob.orders)

        uuid = list(sim.lob.orders.keys())[0]

        sim.set_order(bid=Order(side='bid', price=100 + 2 * sim.order_eps, quantity=100))

        print(sim.lob.orders)  # should have 1 order with different uuid
        assert len(sim.lob.orders) == 1
        assert uuid not in sim.lob.orders


if __name__ == '__main__':
    unittest.main()
