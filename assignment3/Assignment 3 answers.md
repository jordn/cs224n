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
     U: <H x |C|>

  ii)
  Single window
  e(t): O((2w+1)D) operations
  h(t): O((2w+1)D * H + H)
  y(t): O(H*C+C)

  For entire sentence:
  O(H*CT + wDHT + wDT)
  =  O(HCT + wDHT)

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



Q2)
a) i)
Window based model:
E + W + b1 + U + b2
= VD + DH(2w+1) + H + HC + C

RNN:
E + W_h + W_e + b1 + U + b2
= VD + HH + DH + H + HC + C

"How many more params in RNN":    
H^2 - 2wDH

a) ii)
for each time step t:
  e(t): O(D) operations
  h(t): O(H*H + H*D + H)
  y(t): O(H*C+C)

For sentence of length T:
O(T(D + H*H + H*D + H + H*C + C))
=O(TH(H + D + C))
=O(TH(H + D + C))


b) i) E.g. class labels are very uneven for example 99% OTHER, 1% PER.
Just predicting OTHER would lead to a very good cross-entropy loss but would be suboptimal on F1.

ii) It's difficult to directly optimise F1 because F1 is not differentiable and it is calculated 
over the entire corpus.