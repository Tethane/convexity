#include <convexity/core/version.hpp>

#include <gtest/gtest.h>

TEST(ConvexityCore, ExposesVersion) {
    EXPECT_EQ(convexity::version(), "0.1.0");
}
