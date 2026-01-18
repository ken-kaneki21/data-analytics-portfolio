\# Data Dictionary — FD001 Canonical



| Column | Meaning | Type | Notes |

|---|---|---|---|

| engine\_id | Unique engine identifier | int | Entity id |

| cycle | Operating cycle index | int | Time index |

| op\_setting\_1 | Operating setting 1 | float | Unit not specified in dataset |

| op\_setting\_2 | Operating setting 2 | float | Unit not specified in dataset |

| op\_setting\_3 | Operating setting 3 | float | Unit not specified in dataset |

| s1..s21 | Sensor measurements | float | Unit not specified in dataset |

| split | Train/Test split indicator | string | `train` or `test` |

| failure\_cycle | Last observed cycle for engine | int/NA | Filled for train only |

| cycles\_to\_failure | Remaining cycles until failure | int/NA | `failure\_cycle - cycle`, train only |



