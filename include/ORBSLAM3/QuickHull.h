#ifndef BACKEND_QUICKHULL_H
#define BACKEND_QUICKHULL_H

#include <algorithm>
#include <iostream>
#include <vector>

// https://github.com/MiguelVieira/ConvexHull2D/blob/master/ConvexHull.cpp

namespace QuickHull {

struct Point {
    float x;
    float y;

    Point(float xIn, float yIn) : x(xIn), y(yIn) {
    }
};

typedef std::vector<Point> Polygon;

// QuickHull algorithm.
// https://en.wikipedia.org/wiki/QuickHull
std::vector<Point> quickHull(const std::vector<Point> &v);

} // namespace QuickHull

#endif // BACKEND_QUICKHULL_H
