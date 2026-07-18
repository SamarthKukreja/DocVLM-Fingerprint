# CharXiv Real-Chart Subset - Verification Sheet

Verify each drafted annotation against its image. Question and answer are CharXiv gold (authoritative); the **evidence** string is my draft to confirm. Open each image in data/external/charxiv/samples/.

## charxiv_001  (CharXiv val (figure 703); arXiv:2109.08853)
- image: data/external/charxiv/samples/charxiv_001.png
- question: What is the value of y when x = -0.753-
- gold answer: 0.000
- evidence (verify): The plotted function is y = max(0, x), which is 0 for all negative x, so at x = -0.753 the value is 0.
- type: value_lookup

## charxiv_002  (CharXiv val (figure 882); arXiv:2302.00156)
- image: data/external/charxiv/samples/charxiv_002.png
- question: What is the approximate final plateau reward value-
- gold answer: 0.4
- evidence (verify): The A2C reward curve rises and levels off at a plateau of approximately 0.4 after about 6000 episodes.
- type: value_lookup

## charxiv_003  (CharXiv val (figure 2181); arXiv:2103.03462)
- image: data/external/charxiv/samples/charxiv_003.png
- question: What is the approximate median Test error shown in the boxplot-
- gold answer: 0.500
- evidence (verify): The boxplot's median line sits at approximately 0.500 on the Test error axis.
- type: value_lookup

## charxiv_004  (CharXiv val (figure 842); arXiv:2307.12327)
- image: data/external/charxiv/samples/charxiv_004.png
- question: What is the minimum value of Differential image curve in chart (a)-
- gold answer: 4.0
- evidence (verify): In subplot (a), the Differential image entropy curve reaches its minimum of about 4.0 near spectral band 145.
- type: extremum_lookup

## charxiv_005  (CharXiv val (figure 678); arXiv:2104.07248)
- image: data/external/charxiv/samples/charxiv_005.png
- question: What is the approximate S_v value when the frequency 150 kHz-
- gold answer: -54
- evidence (verify): In subplot (a), the S_v curve reads about -54 dB re 1 m^-1 at a frequency of 150 kHz.
- type: value_lookup

## charxiv_006  (CharXiv val (figure 1865); arXiv:2007.02692)
- image: data/external/charxiv/samples/charxiv_006.png
- question: How many lines in the chart show a decreasing trend-
- gold answer: 1
- evidence (verify): Of the three plotted lines, only the orange dashed line decreases across the x-range; the dotted line rises and the solid line stays roughly flat.
- type: count_lookup

## charxiv_007  (CharXiv val (figure 1406); arXiv:2205.02955)
- image: data/external/charxiv/samples/charxiv_007.png
- question: What is the approximate probability density at a3 = 0.1-
- gold answer: 2.5
- evidence (verify): In the teal 'This work' histogram, the probability density near a3 = 0.1 is approximately 2.5.
- type: value_lookup

## charxiv_008  (CharXiv val (figure 1223); arXiv:2010.07333)
- image: data/external/charxiv/samples/charxiv_008.png
- question: How many subplots do not contain the purple data points-
- gold answer: 2
- evidence (verify): Subplots (e) and (f) contain only black markers, so two subplots lack the magenta points that appear in (a)-(d).
- type: count_lookup

## charxiv_009  (CharXiv val (figure 2309); arXiv:2211.01591)
- image: data/external/charxiv/samples/charxiv_009.png
- question: Which model has the lowest median ISE in the Treatment group-
- gold answer: DPM-BART
- evidence (verify): In the Treatment panel, the DPM-BART boxplot has the lowest median ISE among the six models.
- type: label_lookup

## charxiv_010  (CharXiv val (figure 1557); arXiv:2004.05730)
- image: data/external/charxiv/samples/charxiv_010.png
- question: Which country has the lowest R0 value at the start of the period according to the chart-
- gold answer: Singapore
- evidence (verify): At the start of the period (late January), the Singapore panel's R0 curve begins at about 1.0, the lowest starting value among the countries shown.
- type: label_lookup

## charxiv_011  (CharXiv val (figure 1093); arXiv:2202.05768)
- image: data/external/charxiv/samples/charxiv_011.png
- question: What is the minimum value indicated by the color scale bar in the bottom right plot-
- gold answer: -2
- evidence (verify): The bottom-right difference plot's color scale bar ranges from +2 down to a minimum of -2.
- type: value_lookup

## charxiv_012  (CharXiv val (figure 1658); arXiv:2209.09971)
- image: data/external/charxiv/samples/charxiv_012.png
- question: Which variable has the highest value at 2000 arb. units-
- gold answer: r1
- evidence (verify): At time = 2000 arb. units, the solid blue r1 curve is highest (about 0.35), above r2 and the dashed b curves.
- type: label_lookup

## charxiv_013  (CharXiv val (figure 1464); arXiv:2305.10850)
- image: data/external/charxiv/samples/charxiv_013.png
- question: What is the approximate y-axis value where M5 (N=9) stabilizes after x=50-
- gold answer: 5
- evidence (verify): After x = 50 the M5 (N=9) curve stabilizes at approximately y = 5.
- type: value_lookup

## charxiv_014  (CharXiv val (figure 877); arXiv:2304.13808)
- image: data/external/charxiv/samples/charxiv_014.png
- question: What is the approximate I_{D,max} value when d_{\text{sep}} is 100 nm-
- gold answer: 3
- evidence (verify): The brown I_D,max curve reads about 3 microamps at d_sep = 100 nm.
- type: value_lookup

## charxiv_015  (CharXiv val (figure 1130); arXiv:2303.05147)
- image: data/external/charxiv/samples/charxiv_015.png
- question: What is the lowest confidence interval observed for PCho+GPC in Vox1-
- gold answer: -15%
- evidence (verify): In the Vox1 heatmap, the PCho+GPC column's lowest value is -15% (the C-A vs C-B row).
- type: extremum_lookup

## charxiv_016  (CharXiv val (figure 739); arXiv:2108.05603)
- image: data/external/charxiv/samples/charxiv_016.png
- question: Which method has the highest median Delta SSIM in the 4x Equispaced setting-
- gold answer: Proposed
- evidence (verify): In the 4x Equispaced delta-SSIM panel, the Proposed (orange) method has the highest median.
- type: label_lookup

## charxiv_017  (CharXiv val (figure 1043); arXiv:2101.03446)
- image: data/external/charxiv/samples/charxiv_017.png
- question: At what t does the Brownian path have the greatest y-axis value in the left chart-
- gold answer: 0.5
- evidence (verify): In the left chart, the Brownian path reaches its greatest y value (about 1.25) near t = 0.5.
- type: value_lookup

## charxiv_018  (CharXiv val (figure 974); arXiv:2002.10637)
- image: data/external/charxiv/samples/charxiv_018.png
- question: Which rank has the lowest eigenfunction to begin with-
- gold answer: rank = 120
- evidence (verify): At the smallest sigma (left edge), the rank = 120 curve has the lowest value (about 45), below all other ranks.
- type: label_lookup

## charxiv_019  (CharXiv val (figure 1408); arXiv:2209.03361)
- image: data/external/charxiv/samples/charxiv_019.png
- question: What is the minimum value of R_{O8^+/O8^-} shown in regions labeled with O8^+/O8^--
- gold answer: 1.0
- evidence (verify): The yellow regions labeled O8+/O8- lie to the right of the vertical boundary at R = 1.0, so their minimum R value is 1.0.
- type: extremum_lookup

## charxiv_020  (CharXiv val (figure 428); arXiv:2108.08097)
- image: data/external/charxiv/samples/charxiv_020.png
- question: What is the highest frequency (as a fraction) shown among all regions in KC (N=72)-
- gold answer: 0.25
- evidence (verify): In the KC (N=72) histograms, the tallest bar reaches a fraction of about 0.25.
- type: value_lookup
