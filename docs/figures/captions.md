# Figure captions


## Revenue & profit trend

`figures/trend_light.png`

Revenue and profit over time, as shared-x small multiples rather than a dual-axis chart. A dual y-axis would put the crossover point wherever we chose the scales; separate panels keep both measures truthful.


## Geographic distribution

`figures/map_light.png`

Sales by state. Sequential single-hue blue, because sales is a magnitude and more is simply more. Switching the metric to Profit switches the scale to diverging blue/red with a neutral grey midpoint, so a state losing money cannot look like a state making a little.


## Correlation matrix

`figures/correlation_light.png`

Correlation matrix on a diverging scale anchored at zero: -1 must look as strong as +1 while 0 looks like nothing. Cells carry a 2px surface gap.


## Profit distribution

`figures/distribution_light.png`

Profit distribution by category as a box plot rather than a histogram. The question is about spread and the long negative tail; a histogram of a heavily skewed variable hides exactly that.


## Category composition

`figures/sunburst_light.png`

Sales composition, Category to Sub-Category. The hierarchy is genuine - each sub-category belongs to exactly one category - which is the precondition for a sunburst being honest rather than decorative.


## Animated year-over-year

`figures/animated_light.png`

Sales by region animated across years. The y-axis is fixed across all frames; letting it rescale per frame would make every year look identical and destroy the comparison the animation exists to make.


## Discount vs profit regression

`figures/scatter_light.png`

Discount against profit, with a least-squares fit and a 95% confidence interval on the mean response. This is the project's central finding: margin collapses as discount deepens.


## Segment & category breakdown

`figures/stacked_light.png`

Sales by customer segment, stacked by category, with a 2px surface gap between segments so adjacent fills stay distinguishable.
