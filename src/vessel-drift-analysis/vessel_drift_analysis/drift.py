"""Probability that a vessel goes adrift.

The probability that a vessel goes adrift is estimated using data derived from the
Vessels of Concern database provided by the Marine Exchange and an estimate of total
vessels in the domain as derived from AIS data.

The database includes Vessels of Concern database includes 193 instances of drifting
drifting vessels from 2015-01-15 to 2019-10-18.

The number of vessels were estimated by counting the number of voyages in the satellite
based AIS data for the year 2019 (the same year as the forcings).

.. code-block:: bash

    > for f in $AIS_DATA_DIR/*.csv; do wc -l $f >> voyages.txt; done
    > awk '{ sum += $1 } END { print sum; }' voyages.txt > total_voyages_2019.txt 

The estimated probability is the ratio of the number of drifting vessels to the number
of voyages.  That probability is then converted to a daily value by dividing by the
number of days in the database.  That is, the probability used in the analysis is the
the estimated probability that a vessel will drift per day.

Resources:
----------
..[1] https://www.mxak.org/services/mda/tracking/
"""
import datetime

N_VESSELS_OF_CONCERN = 193
N_VOYAGES = 281771
START_DATE = datetime.date(2015, 1, 15)
END_DATE = datetime.date(2019, 10, 18)


def calculate_drift_probability(
    n_vessels_of_concern: int = N_VESSELS_OF_CONCERN,
    n_voyages: int = N_VOYAGES,
    start_date: datetime.datetime = START_DATE,
    end_date: datetime.datetime = END_DATE
) -> float:
    """Return the estimated daily probability that a vessel goes adrift.

    Parameters
    ----------
    start_date : datetime.date
        The start date of the analysis.
    end_date : datetime.date
        The end date of the analysis.

    Returns
    -------
    float
        The estimated probability that a vessel goes adrift.
    """
    n_days = (end_date - start_date).days
    n_voyages_per_day = n_voyages / n_days
    n_vessels_of_concern_per_day = n_vessels_of_concern / n_days

    return n_vessels_of_concern_per_day / n_voyages_per_day
