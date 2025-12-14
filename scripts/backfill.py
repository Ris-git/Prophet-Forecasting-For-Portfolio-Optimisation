
import logging
import sys
import pandas as pd
from datetime import timedelta
from src.main import run_optimisation
from src.database import save_results_to_supabase
from src.settings import PORTFOLIO_TICKERS, START_DATE

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def backfill(days: int = 5):
    """
    Backfill prediction data for the last 'days' business days.
    """
    # Get last N business days, excluding today (since we might have already run today)
    # Actually, let's include today if needed, but the user just ran today.
    # Let's run for the *previous* 5 days to ensure ample history.
    
    end_date = pd.Timestamp.now().normalize()
    # Generate business dates ending yesterday
    dates = pd.bdate_range(end=end_date - timedelta(days=1), periods=days)
    
    logger.info(f"Starting backfill for dates: {[d.strftime('%Y-%m-%d') for d in dates]}")

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"--------------------------------------------------")
        logger.info(f"Processing backfill for date: {date_str}")
        logger.info(f"--------------------------------------------------")
        
        try:
            # We keep the same START_DATE (2020-01-01) but vary the END_DATE
            # This simulates "what would the model have predicted on that day?"
            result = run_optimisation(
                tickers=PORTFOLIO_TICKERS, 
                start_date=START_DATE, 
                end_date=date_str
            )
            
            if result:
                save_results_to_supabase(result)
                logger.info(f"Successfully saved backfill data for {date_str}")
            else:
                logger.warning(f"No result generated for {date_str}")
                
        except Exception as e:
            logger.error(f"Failed backfill for {date_str}: {e}")
            # Continue to next date even if one fails
            continue

if __name__ == "__main__":
    # Check if a number of days is provided as arg
    days_to_backfill = 5
    if len(sys.argv) > 1:
        try:
            days_to_backfill = int(sys.argv[1])
        except ValueError:
            pass
            
    backfill(days=days_to_backfill)
