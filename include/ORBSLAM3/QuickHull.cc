#include <algorithm>
#include <vector>
#include <cmath>
#include "QuickHull.h"

namespace QuickHull {

using namespace std;

// The z-value of the cross product of segments
// (a, b) and (a, c). Positive means c is ccw
// from (a, b), negative cw. Zero means its collinear.
float ccw(const Point &a, const Point &b, const Point &c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// Returns true if a is lexicographically before b.
bool isLeftOf(const Point &a, const Point &b) {
    return (a.x < b.x || (a.x == b.x && a.y < b.y));
}

// Used to sort points in ccw order about a pivot.
struct ccwSorter {
    const Point &pivot;

    ccwSorter(const Point &inPivot) : pivot(inPivot) {
    }

    bool operator()(const Point &a, const Point &b) {
        return ccw(pivot, a, b) < 0;
    }
};

// The length of segment (a, b).
float len(const Point &a, const Point &b) {
    return std::sqrt((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y));
}

// The unsigned distance of p from segment (a, b).
float dist(const Point &a, const Point &b, const Point &p) {
    return std::fabs((b.x - a.x) * (a.y - p.y) - (b.y - a.y) * (a.x - p.x)) /
           len(a, b);
}

// Returns the index of the farthest Point from segment (a, b).
size_t getFarthest(const Point &a, const Point &b, const vector<Point> &v) {
    size_t idxMax = 0;
    float distMax = dist(a, b, v[idxMax]);

    for (size_t i = 1; i < v.size(); ++i) {
        float distCurr = dist(a, b, v[i]);
        if (distCurr > distMax) {
            idxMax = i;
            distMax = distCurr;
        }
    }

    return idxMax;
}

// The gift-wrapping algorithm for convex hull.
// https://en.wikipedia.org/wiki/Gift_wrapping_algorithm
vector<Point> giftWrapping(vector<Point> v) {
    // Move the leftmost Point to the beginning of our vector.
    // It will be the first Point in our convext hull.
    swap(v[0], *min_element(v.begin(), v.end(), isLeftOf));

    vector<Point> hull;
    // Repeatedly find the first ccw Point from our last hull Point
    // and put it at the front of our array.
    // Stop when we see our first Point again.
    do {
        hull.push_back(v[0]);
        swap(v[0], *min_element(v.begin() + 1, v.end(), ccwSorter(v[0])));
    } while (v[0].x != hull[0].x && v[0].y != hull[0].y);

    return hull;
}

// The Graham scan algorithm for convex hull.
// https://en.wikipedia.org/wiki/Graham_scan
vector<Point> GrahamScan(vector<Point> v) {
    // Put our leftmost Point at index 0
    swap(v[0], *min_element(v.begin(), v.end(), isLeftOf));

    // Sort the rest of the points in counter-clockwise order
    // from our leftmost Point.
    sort(v.begin() + 1, v.end(), ccwSorter(v[0]));

    // Add our first three points to the hull.
    vector<Point> hull;
    auto it = v.begin();
    hull.push_back(*it++);
    hull.push_back(*it++);
    hull.push_back(*it++);

    while (it != v.end()) {
        // Pop off any points that make a convex angle with *it
        while (ccw(*(hull.rbegin() + 1), *(hull.rbegin()), *it) >= 0) {
            hull.pop_back();
        }
        hull.push_back(*it++);
    }

    return hull;
}

// The monotone chain algorithm for convex hull.
vector<Point> monotoneChain(vector<Point> v) {
    // Sort our points in lexicographic order.
    sort(v.begin(), v.end(), isLeftOf);

    // Find the lower half of the convex hull.
    vector<Point> lower;
    for (auto it = v.begin(); it != v.end(); ++it) {
        // Pop off any points that make a convex angle with *it
        while (lower.size() >= 2 &&
               ccw(*(lower.rbegin() + 1), *(lower.rbegin()), *it) >= 0) {
            lower.pop_back();
        }
        lower.push_back(*it);
    }

    // Find the upper half of the convex hull.
    vector<Point> upper;
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
        // Pop off any points that make a convex angle with *it
        while (upper.size() >= 2 &&
               ccw(*(upper.rbegin() + 1), *(upper.rbegin()), *it) >= 0) {
            upper.pop_back();
        }
        upper.push_back(*it);
    }

    vector<Point> hull;
    hull.insert(hull.end(), lower.begin(), lower.end());
    // Both hulls include both endpoints, so leave them out when we
    // append the upper hull.
    hull.insert(hull.end(), upper.begin() + 1, upper.end() - 1);
    return hull;
}

// Recursive call of the quickhull algorithm.
void quickHull(
    const vector<Point> &v, const Point &a, const Point &b,
    vector<Point> &hull) {
    if (v.empty()) {
        return;
    }

    Point f = v[getFarthest(a, b, v)];

    // Collect points to the left of segment (a, f)
    vector<Point> left;
    for (auto p : v) {
        if (ccw(a, f, p) > 0) {
            left.push_back(p);
        }
    }
    quickHull(left, a, f, hull);

    // Add f to the hull
    hull.push_back(f);

    // Collect points to the left of segment (f, b)
    vector<Point> right;
    for (auto p : v) {
        if (ccw(f, b, p) > 0) {
            right.push_back(p);
        }
    }
    quickHull(right, f, b, hull);
}

// QuickHull algorithm.
// https://en.wikipedia.org/wiki/QuickHull
vector<Point> quickHull(const vector<Point> &v) {
    vector<Point> hull;

    // Start with the leftmost and rightmost points.
    Point a = *min_element(v.begin(), v.end(), isLeftOf);
    Point b = *max_element(v.begin(), v.end(), isLeftOf);

    // Split the points on either side of segment (a, b)
    vector<Point> left, right;
    for (auto p : v) {
        ccw(a, b, p) > 0 ? left.push_back(p) : right.push_back(p);
    }

    // Be careful to add points to the hull
    // in the correct order. Add our leftmost Point.
    hull.push_back(a);

    // Add hull points from the left (top)
    quickHull(left, a, b, hull);

    // Add our rightmost Point
    hull.push_back(b);

    // Add hull points from the right (bottom)
    quickHull(right, b, a, hull);

    return hull;
}

} // namespace QuickHull
