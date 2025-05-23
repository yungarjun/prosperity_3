from ast import And
import json
from typing import Any, List
import string
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import jsonpickle

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

# Updated parameters for three products:
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,          
        "clear_width": 1,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 9,
    },
    Product.KELP: {
        "fair_value": 2032,
        "take_width": 1,
        "clear_width": 2,
        "prevent_adverse": False,
        "adverse_volume": 1,
        "reversion_beta": -0.6,
        "disregard_edge": 2,
        "join_edge": 0.5,
        "default_edge": 1,
        "soft_position_limit": 9,
    },
    Product.SQUID_INK: {
        "fair_value": 2000,         # Baseline fair value (starting point)
        "take_width": 0.5,          
        "clear_width": 1,           
        "prevent_adverse": False,
        "adverse_volume": 20,
        "reversion_beta": -0.25,    # Used in mean reverting mode
        "breakout_threshold": 50,   # If midprice deviates more than 30 units, follow the trend
        "disregard_edge": 0.5,
        "join_edge": 0.5,
        "default_edge": 1,
        "soft_position_limit": 50,
    },
}

class Trader:
    def __init__(self, params=None, flush=True) -> None:
        if params is None:
            params = PARAMS
        self.params = params
        self.flush = flush

        # Set maximum order amounts for the three products
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
        }

    def take_best_orders(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        # Sell side: if best ask is below fair_value - take_width, execute a buy.
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or best_ask_amount <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # Buy side: if best bid is above fair_value + take_width, execute a sell.
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or best_bid_amount <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price for price in order_depth.sell_orders.keys() if price >= fair_value + disregard_edge
        ]
        bids_below_fair = [
            price for price in order_depth.buy_orders.keys() if price <= fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )

        return orders, buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:

        '''
        Compute fair value from order depth using bid and ask from market maker
        '''

        if order_depth.sell_orders and order_depth.buy_orders:
            max_ask = max(order_depth.sell_orders.keys())
            min_bid = min(order_depth.buy_orders.keys())

            if max_ask and min_bid:
                # Calculate the fair value as the average of the best ask and best bid
                fair_value = (max_ask + min_bid) / 2
                traderObject["kelp_last_price"] = fair_value
                return fair_value
            else:
                if traderObject.get("kelp_last_price") is not None:
                    return traderObject["kelp_last_price"]
                else:
                    return None
        else:
            return None

        # """
        # Compute a dynamic, mean-reverting fair value for KELP.
        # """

        # if order_depth.sell_orders and order_depth.buy_orders:
        #     best_ask = min(order_depth.sell_orders.keys())
        #     best_bid = max(order_depth.buy_orders.keys())
        #     filtered_ask = [
        #         price for price in order_depth.sell_orders.keys()
        #         if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
        #     ]
        #     filtered_bid = [
        #         price for price in order_depth.buy_orders.keys()
        #         if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
        #     ]
        #     mm_ask = min(filtered_ask) if filtered_ask else None
        #     mm_bid = max(filtered_bid) if filtered_bid else None
        #     if mm_ask is None or mm_bid is None:
        #         if traderObject.get("kelp_last_price") is None:
        #             mmmid_price = (best_ask + best_bid) / 2
        #         else:
        #             mmmid_price = traderObject["kelp_last_price"]
        #     else:
        #         mmmid_price = (mm_ask + mm_bid) / 2

        #     if traderObject.get("kelp_last_price") is not None:
        #         last_price = traderObject["kelp_last_price"]
        #         last_returns = (mmmid_price - last_price) / last_price
        #         pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
        #         fair = mmmid_price + (mmmid_price * pred_returns)
        #     else:
        #         fair = mmmid_price

        #     traderObject["kelp_last_price"] = mmmid_price
        #     return fair
        # return None

    def squidink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        """
        Hybrid fair value for SQUID_INK: If the midprice deviates beyond a breakout threshold
        from the baseline, follow the trend. Otherwise, apply a mean-reverting adjustment.
        """
        params = self.params[Product.SQUID_INK]
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return params["fair_value"]
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        midprice = (best_ask + best_bid) / 2
        static_fv = params["fair_value"]
        breakout_threshold = params["breakout_threshold"]

        # Check for breakout/trending mode:
        if midprice > static_fv + breakout_threshold:
            fv = midprice  # Follow upward trend
        elif midprice < static_fv - breakout_threshold:
            fv = midprice  # Follow downward trend
        else:
            # Mean reverting adjustment
            if traderObject.get("squidink_last_price") is None:
                mmmid_price = midprice
            else:
                mmmid_price = traderObject["squidink_last_price"]
            if traderObject.get("squidink_last_price") is not None:
                last_price = traderObject["squidink_last_price"]
                last_returns = (midprice - last_price) / last_price
                fv = midprice + (midprice * last_returns * params["reversion_beta"])
            else:
                fv = midprice

        traderObject["squidink_last_price"] = midprice
        return fv

    def run(self, state: TradingState):
        # Load or initialize persistent trader data.
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # Process RAINFOREST_RESIN with its static fair value strategy.
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                resin_position,
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )

        # Process KELP using the dynamic mean-reverting fair value.
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_fair = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject)
            if kelp_fair is not None:
                kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
                kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
                kelp_make_orders, _, _ = self.make_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair,
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.KELP]["disregard_edge"],
                    self.params[Product.KELP]["join_edge"],
                    self.params[Product.KELP]["default_edge"],
                    True,
                    self.params[Product.KELP]["soft_position_limit"],
                )
                result[Product.KELP] = (
                    kelp_take_orders + kelp_clear_orders + kelp_make_orders
                )

        # Process SQUID_INK using the hybrid fair value (trend-following in breakout mode,
        # and mean reverting otherwise).
        if Product.SQUID_INK in state.order_depths:
            squidink_position = state.position.get(Product.SQUID_INK, 0)
            squidink_fair = self.squidink_fair_value(state.order_depths[Product.SQUID_INK], traderObject)
            if squidink_fair is not None:
                squidink_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squidink_fair,
                    self.params[Product.SQUID_INK]["take_width"],
                    squidink_position,
                    self.params[Product.SQUID_INK].get("prevent_adverse", False),
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
                squidink_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squidink_fair,
                    self.params[Product.SQUID_INK]["clear_width"],
                    squidink_position,
                    buy_order_volume,
                    sell_order_volume,
                )
                squidink_make_orders, _, _ = self.make_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squidink_fair,
                    squidink_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.SQUID_INK]["disregard_edge"],
                    self.params[Product.SQUID_INK]["join_edge"],
                    self.params[Product.SQUID_INK]["default_edge"],
                    True,
                    self.params[Product.SQUID_INK]["soft_position_limit"],
                )
                result[Product.SQUID_INK] = (
                    squidink_take_orders + squidink_clear_orders + squidink_make_orders
                )

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        if self.flush:
            logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
