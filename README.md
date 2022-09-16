# KDE
Intership Trace. Using KDE so display tool wear based on current fluctuations.

Using JSON as entry with platform logs, specified tools and bandwiths to work on.
(specify "None" in "bandwith" will trigger a GridSearch to fine the most optimal bandwidth to the given tool)

The scipt will give you a directory for each tool specified and for each bandwiths specified, thus you will be able to navitage throught every output of the script on the HTLM output (note that the HTML is test-file and the final version remains to the company)

We did try a lot a thing, such as Gaussian Mixture (ND knn) or manual KDE, to get this final version.
