Q1)
a) i)
- WHO spent £2 million on aid last year - either WHO (ORG) or who (other)
- The Bill and Melinda Gates foundation - Melinda Gates (PERS) or Bill and Melinda Gates Foundation (ORG)

ii) The context can help distinguish which entity it is.

iii) Context words / Context entities / capitalisation / Gazetteers (NER lookups)

b)

 i)
 With window size w, embedding dim D
     e(t): <1 x (2w+1)D>
     W: <(2w+1)*D x H>
     U: <H x |V|>

  ii)
  Single window
  e(t): O((2w+1)D) operations
  h(t): O((2w+1)D * H + H)
  y(t): O(H*V+V)

  For entire sentence:
  O(H*VT + wDHT + wDT)
  =  O(HVT + wDHT)

d)
i)

Entity level P/R/F1: 0.81/0.85/0.83
Token-level confusion matrix:
gold\guess  PER     	ORG     	LOC     	MISC    	O
PER     	2927.00 	76.00   	69.00   	12.00   	65.00
ORG     	126.00  	1662.00 	116.00  	64.00   	124.00
LOC     	36.00   	128.00  	1873.00 	15.00   	42.00
MISC    	34.00   	68.00   	50.00   	1008.00 	108.00
O       	35.00   	56.00   	18.00   	28.00   	42622.00


Most data is Other. Mostly confuse Orgs for Locs.

ii)

Doesn't really pay attention to surrounding words type. Very unlikely to be Org Loc.
Leads to non-contiguous labels.
ie.

hey there Jordan Burgess
O   O     PER    ORG





