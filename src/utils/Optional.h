#pragma once

template <typename T>
class Optional {
private:
    T _value;
    bool _existed;

public:
    Optional(): _existed{false} {}
    Optional(const T&& elem): _value{elem}, _existed{false} {}

    Optional<T>& operator=(T elem) {
        _value = elem;
        _existed = true;
        return *this;
    }

    bool has_value() {
        return _existed;
    }

    T value() {
        return _value;
    }

    T value_or(T elem) {
        return _existed? _value: elem;
    }
};