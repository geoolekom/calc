//
// Created by geoolekom on 23.02.19.
//

#ifndef CALC_SPACE_H
#define CALC_SPACE_H

#include <array>
#include <iostream>


template <typename T, std::size_t dimension>
T operator*(const std::array<T, dimension> one, const std::array<T, dimension> two) {
    T retVal = 0;
    for (std::size_t i = 0; i < dimension; i ++) {
        retVal += one[i] * two[i];
    }
    return retVal;
}

template <typename T, std::size_t dimension>
std::array<T, dimension> operator+(const std::array<T, dimension> one, const std::array<T, dimension> two) {
    std::array<T, dimension> ret;
    for (std::size_t i = 0; i < dimension; i ++) {
        ret[i] = one[i] + two[i];
    }
    return ret;
}

template <typename T, std::size_t dimension>
std::array<T, dimension> operator-(const std::array<T, dimension> one, const std::array<T, dimension> two) {
    std::array<T, dimension> ret;
    for (std::size_t i = 0; i < dimension; i ++) {
        ret[i] = one[i] - two[i];
    }
    return ret;
}

template <std::size_t dimension>
double operator*(const std::array<double, dimension> one, const std::array<int, dimension> two) {
    double ret = 0;
    for (std::size_t i = 0; i < dimension; i ++) {
        ret += one[i] * two[i];
    }
    return ret;
}

template <typename T, std::size_t dimension>
std::array<T, dimension> operator*(const std::array<std::array<T, dimension>, dimension>& m, const std::array<T, dimension> v) {
    std::array<T, dimension> ret;
    for (std::size_t i = 0; i < dimension; i ++) {
        ret[i] = m[i] * v;
    }
    return ret;
}

template <typename T, std::size_t dimension>
std::ostream& operator<<(std::ostream& out, const std::array<T, dimension> v) {
    for (const auto& e : v) {
        out << e << "\t";
    }
    return out;
}


template <std::size_t dimension>
class Iterable {
public:
    typedef std::array<int, dimension> Vector;
private:
    const Vector bounds;

    class Iterator {
    private:
        Vector value = {{ 0 }};
        int counter = 0;
        const Vector bounds;
    public:
        Iterator(int counter, const Vector& bounds) : counter(counter), bounds(bounds) {
            this->recount();
        };

        Vector& operator*() { return value; };

        Iterator& operator++() {
            this->counter ++;
            this->recount();
            return *this;
        }

        bool operator==(const Iterator& other) { return this->counter == other.counter; }
        bool operator!=(const Iterator& other) { return this->counter != other.counter; }
        bool operator<(const Iterator& other) { return this->counter < other.counter; }

        void recount() {
            int localCounter = counter;
            for (int i = dimension - 1; i >= 0; i --) {
                value[i] = localCounter % bounds[i];
                localCounter /= bounds[i];
            }
        }
    };

public:
    Iterable() = delete;

    Iterable(const Vector& bounds) : bounds(bounds) {};

    Iterator begin() {
        return {0, bounds};
    }

    Iterator end() {
        int finalCounter = 1;
        for (const auto& e : bounds) {
            finalCounter *= e;
        }
        return {finalCounter, bounds};
    }
};


template <std::size_t dimension>
class Space {
public:
    typedef Space<dimension> selfType;
    typedef Iterable<dimension> iterableType;
    typedef typename iterableType::Vector vectorType;
private:
    double* data;
    int xSize = 1, vSize = 1;
public:
    const vectorType xBounds, vBounds;

    Space(const vectorType& xBounds, const vectorType& vBounds) : xBounds(xBounds), vBounds(vBounds) {
        for (const auto& e: xBounds) {
            xSize *= e;
        }
        for (const auto& e: vBounds) {
            vSize *= e;
        }
        data = new double[xSize * vSize]();
    };

    Space(const selfType& s) : Space(s.xBounds, s.vBounds) {};

    ~Space() {
        delete[] data;
    }

    double* getData() const {
        return this->data;
    }

    int getSize() const {
        return this->xSize * this->vSize;
    }

    iterableType spaceIterable() {
        return {xBounds};
    }

    iterableType velocityIterable() {
        return {vBounds};
    };

    double* velocitySlice(const vectorType& xIndex) {
        return this->getData() + vSize * (xIndex * xBounds);
    };

    double getValue(const vectorType& xIndex, const vectorType& vIndex) {
        return this->getData()[index(xIndex, vIndex)];
    };

    void setValue(const vectorType& xIndex, const vectorType& vIndex, double value) {
        this->getData()[index(xIndex, vIndex)] = value;
    };

    inline int index(const vectorType& xIndex, const vectorType& vIndex) {
        int index = 0;
        int multiplier = 1;
        for (int i = 0; i < dimension; i ++) {
            index += ((vIndex[i] + vBounds[i]) % vBounds[i]) * multiplier;
            multiplier *= vBounds[i];
        }
        for (int i = 0; i < dimension; i ++) {
            index += ((xIndex[i] + xBounds[i]) % xBounds[i]) * multiplier;
            multiplier *= xBounds[i];
        }
        return index;
    }
};


#endif //CALC_SPACE_H
