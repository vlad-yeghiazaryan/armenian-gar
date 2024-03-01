* ReadMe_Main.txt
* Last Updated: April 13, 2023

==================================================================
I. The IMF's integrated macroprudential policy (iMaPP) database
==================================================================
Citation. When using this data for your research or article, 
we would appreciate it if you could kindly acknowledge that
“the IMF’s integrated Macroprudential Policy (iMaPP) Database, 
originally constructed by Alam et al. (2019)” as the source information, 
citing the following paper: 
  Alam, Zohair, Adrian Alter, Jesse Eiseman, Gaston Gelos, Heedon Kang, 
  Machiko Narita, Erlend Nier, and Naixi Wang (2019) 
  "Digging Deeper – Evidence on the Effects of Macroprudential Policies 
  from a New Database", IMF Working Paper No. 19/66.

==================================================================
II. Contents of this zip file
==================================================================
1. iMaPP_database-2023-4-11.xlsx
	The iMaPP database. Please see the TOR sheet in the file 
	for the contents. Please note that the iMaPP database will 
	be annually updated by the IMF and the file name indicates 
	its version. Please also see "Notes" below.	

2. iMaPP_load.do
	The STATA code to load indicators from the iMaPP database
	and save them in the STATA data format (i.e., in a DTA file).

3. Alam et al. (2019) iMaPP WP.pdf
	The IMF working paper that introduced the iMaPP database,
	which is available at: www.imf.org/iMaPP.

4. Subfolder: "Sample Files for Figures in Alam et al. (2019)"
   This folder contains sample Stata codes and Excel files to 
   demonstrate how to generate descriptive charts, similar to 
   Fig 1-5 in Alam et al. (2019).

   4.1. Stata Codes:
    (a) iMaPP_Fig123.do for Figures 1, 2, and 3
    (b) iMaPP_Fig4.do for Figure 4
    (c) iMaPP_Fig5.do for Figure 5

   4.2. Excel file:
    (a) iMaPP_Fig1234.xlsx

   4.3. EMF files:
    (a) iMaPP_Fig5a.emf and (b) iMaPP_Fig5b.emf for Figure 5

   4.4. Word file: 
    (a) iMaPP_LTV_average_statistics.docx for summary stats 
        of the average LTV limit

==================================================================
III. Notes
==================================================================
April 13, 2023: The 4th update (iMaPP_database-2023-4-11.xlsx) extended 
  the data through 2021M12, refining the data for 2020. Please note that 
  some historical corrections were also made, reflecting additional 
  information that became available from the IMF survey.

January 6, 2022: The 3rd update (iMaPP_database-2022-1-5.xlsx) extended 
  the data through 2020M12, refining the data for 2019. Please note that 
  some historical corrections were also made, reflecting additional 
  information that became available from the IMF survey.

August 31, 2021: The 2nd update (iMaPP_database-2021-8-26.xlsx) extended 
  the data through 2019M12, refining the data for 2018. Please note that 
  some historical corrections were also made, reflecting additional 
  information that became available, including LTV_average for Brazil.
  LTV_median was also added as supplemental information for LTV_average.

September 8, 2020: The 1st update (iMaPP_database -- 2020-09-08.xlsx) 
  extended the data through 2018M12. Please note that comprehensive 
  historical revisions were also made to ensure consistency between 
  the iMaPP and IMF survey; and the numerical indicator of LTV limits 
  (LTV_average) was also revamped.

END