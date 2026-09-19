# Data

The raw file is **not committed**. Download it with:

```bash
make data        # or: python -m rental.data
```

**Source.** ~12.4k Tehran rental listings (deposit, monthly rent, area, build year, amenities, neighborhood)
originally collected from public Divar (divar.ir) ads. The CSV is downloaded from
<https://github.com/amiralimadadi/Regression_TheranHousing> (`Data.csv`).
Please check the original repository / Divar's terms before redistributing the data.

**Columns (raw):** `total_value, neighborhood, area, year, deposit, rent, elavator, parking, warehouse`

| Column | Meaning |
|---|---|
| `neighborhood` | Neighborhood name (Persian) |
| `area` | Floor area, m² |
| `year` | Build year, Jalali calendar |
| `deposit` | Deposit / mortgage amount (ودیعه / رهن), Toman |
| `rent` | Monthly rent, Toman (0 = full-deposit listing) |
| `elavator`, `parking`, `warehouse` | 0/1 amenities (`warehouse` is ~constant and dropped) |
| `total_value` | Provided by the source but is almost equal to `deposit`; **not used** |

All monetary values are asking prices in Toman from roughly 1399-1400 (2020-2021).
