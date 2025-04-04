# MetaTrader 5 MCP Server

A Model Context Protocol (MCP) server for interacting with the MetaTrader 5 trading platform. This server provides AI assistants with tools and resources to access market data, perform trading operations, and analyze trading history.

## Features

- **Market Data Access**: Fetch historical price data, ticks, and symbol information
- **Trading Operations**: Place orders, manage positions, and track trading history
- **Account Management**: Access account information and terminal status
- **AI-Friendly Prompts**: Pre-defined conversation templates for common trading tasks
- **Comprehensive Documentation**: Detailed guides for using the MetaTrader 5 API

## Prerequisites

- Python 3.8 or higher
- MetaTrader 5 terminal installed
- MetaTrader 5 account (demo or real)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the MetaTrader 5 terminal is installed and running.

## Usage

### Running the Server in Development Mode

To run the server in development mode with hot reloading:

```bash
fastmcp dev main.py
```

This will start the MCP server and provide an inspector interface for testing the tools and resources.

### Installing the Server for Claude Desktop

To install the server for use with Claude Desktop:

```bash
fastmcp install main.py
```

### Using the Server with AI Assistants

Once the server is running, AI assistants can use the following tools and resources:

#### Connection Management

- `initialize()`: Initialize the MetaTrader 5 terminal
- `login(account, password, server)`: Log in to a trading account
- `shutdown()`: Shut down the connection to the terminal

#### Market Data Access

- `get_symbols()`: Get all available symbols
- `get_symbol_info(symbol)`: Get information about a specific symbol
- `copy_rates_from_pos(symbol, timeframe, start_pos, count)`: Get historical price data
- `copy_ticks_from_pos(symbol, start_pos, count, flags)`: Get historical tick data

#### Trading Operations

- `order_send(request)`: Send a trading order
- `positions_get(symbol, group)`: Get open positions
- `orders_get(symbol, group)`: Get active orders
- `history_orders_get(...)`: Get historical orders
- `history_deals_get(...)`: Get historical deals

## Example Workflows

### Connecting and Getting Market Data

```python
# Initialize MT5
initialize()

# Log in to your trading account
login(account=123456, password="your_password", server="your_server")

# Get available symbols
symbols = get_symbols()

# Get recent price data for EURUSD
rates = copy_rates_from_pos(symbol="EURUSD", timeframe=15, start_pos=0, count=100)

# Shut down the connection
shutdown()
```

### Placing a Trade

```python
# Initialize and log in
initialize()
login(account=123456, password="your_password", server="your_server")

# Create an order request
request = OrderRequest(
    action=mt5.TRADE_ACTION_DEAL,
    symbol="EURUSD",
    volume=0.1,
    type=mt5.ORDER_TYPE_BUY,
    price=1.1,
    deviation=20,
    magic=123456,
    comment="Buy order",
    type_time=mt5.ORDER_TIME_GTC,
    type_filling=mt5.ORDER_FILLING_IOC
)

# Send the order
result = order_send(request)

# Shut down the connection
shutdown()
```

## Resources

The server provides the following resources to help AI assistants understand how to use the MetaTrader 5 API:

- `mt5://getting_started`: Basic workflow for using the MetaTrader 5 API
- `mt5://trading_guide`: Guide for placing and managing trades
- `mt5://market_data_guide`: Guide for accessing and analyzing market data
- `mt5://timeframes`: Information about available timeframes
- `mt5://tick_flags`: Information about tick flags
- `mt5://order_types`: Information about order types
- `mt5://order_filling_types`: Information about order filling types
- `mt5://order_time_types`: Information about order time types
- `mt5://trade_actions`: Information about trade request actions

## Prompts

The server provides the following prompts to help AI assistants interact with users:

- `connect_to_mt5(account, password, server)`: Connect to MetaTrader 5 and log in
- `analyze_market_data(symbol, timeframe)`: Analyze market data for a specific symbol
- `place_trade(symbol, order_type, volume)`: Place a trade for a specific symbol
- `manage_positions()`: Manage open positions
- `analyze_trading_history(days)`: Analyze trading history

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [MetaQuotes](https://www.metaquotes.net/) for the MetaTrader 5 platform
- [FastMCP](https://github.com/jlowin/fastmcp) for the MCP server implementation
