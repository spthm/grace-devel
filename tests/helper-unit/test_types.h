#pragma once

#define NUM_TEST_TYPES 9

struct Byte1T {
    char data[1];

    GRACE_HOST_DEVICE Byte1T(int init) {
        data[0] = init & 0xff;
    }
};

struct Byte2T {
    char data[2];

    GRACE_HOST_DEVICE Byte2T(int init) {
        data[0] = init & 0xff;
        data[1] = init & 0xff00;
    }
};

struct Byte3T {
    char data[3];

    GRACE_HOST_DEVICE Byte3T(int init) {
        data[0] = init & 0xff;
        data[1] = init & 0xff00;
        data[2] = init & 0xff0000;
    }
};

struct Byte4T {
    char data[4];

    GRACE_HOST_DEVICE Byte4T(int init) {
        data[0] = init & 0xff;
        data[1] = init & 0xff00;
        data[2] = init & 0xff0000;
        data[3] = init & 0xff000000;
    }
};

struct Byte5T {
    char data[5];

    GRACE_HOST_DEVICE Byte5T(int init) {
        data[0] = init & 0xff;
        data[1] = init & 0xff00;
        data[2] = init & 0xff0000;
        data[3] = init & 0xff000000;
        data[4] = (char)0;
    }
};

struct Byte6T {
    char data[6];

    GRACE_HOST_DEVICE Byte6T(int init) {
        data[0] = init & 0xff;
        data[1] = init & 0xff00;
        data[2] = init & 0xff0000;
        data[3] = init & 0xff000000;
        data[4] = (char)0;
        data[5] = (char)0;
    }
};

struct Byte7T {
    char data[7];

    GRACE_HOST_DEVICE Byte7T(int init) {
        data[0] = init & 0xff;
        data[1] = init & 0xff00;
        data[2] = init & 0xff0000;
        data[3] = init & 0xff000000;
        data[4] = (char)0;
        data[5] = (char)0;
        data[6] = (char)0;
    }
};

struct Byte8T {
    char data[8];

    GRACE_HOST_DEVICE Byte8T(int init) {
        data[0] = init & 0xff;
        data[1] = init & 0xff00;
        data[2] = init & 0xff0000;
        data[3] = init & 0xff000000;
        data[4] = (char)0;
        data[5] = (char)0;
        data[6] = (char)0;
        data[7] = (char)0;
    }
};

struct Byte16T {
    char data[16];

    GRACE_HOST_DEVICE Byte16T(int init) {
        data[0] = init & 0xff;
        data[1] = init & 0xff00;
        data[2] = init & 0xff0000;
        data[3] = init & 0xff000000;
        data[4] = (char)0;
        data[5] = (char)0;
        data[6] = (char)0;
        data[7] = (char)0;
        data[8] = (char)0;
        data[9] = (char)0;
        data[10] = (char)0;
        data[11] = (char)0;
        data[12] = (char)0;
        data[13] = (char)0;
        data[14] = (char)0;
        data[15] = (char)0;
    }
};

GRACE_HOST_DEVICE bool operator==(const Byte1T& lhs, const Byte1T& rhs)
{
    return lhs.data[0] == rhs.data[0];
}

GRACE_HOST_DEVICE bool operator==(const Byte2T& lhs, const Byte2T& rhs)
{
    return (lhs.data[0] == rhs.data[0]) && (lhs.data[1] == rhs.data[1]);
}

GRACE_HOST_DEVICE bool operator==(const Byte3T& lhs, const Byte3T& rhs)
{
    return (lhs.data[0] == rhs.data[0]) && (lhs.data[1] == rhs.data[1])
             && (lhs.data[2] == rhs.data[2]);
}

GRACE_HOST_DEVICE bool operator==(const Byte4T& lhs, const Byte4T& rhs)
{
    return (lhs.data[0] == rhs.data[0]) && (lhs.data[1] == rhs.data[1])
             && (lhs.data[2] == rhs.data[2]) && (lhs.data[3] == rhs.data[3]);
}

GRACE_HOST_DEVICE bool operator==(const Byte5T& lhs, const Byte5T& rhs)
{
    return (lhs.data[0] == rhs.data[0]) && (lhs.data[1] == rhs.data[1])
             && (lhs.data[2] == rhs.data[2]) && (lhs.data[3] == rhs.data[3]);
}

GRACE_HOST_DEVICE bool operator==(const Byte6T& lhs, const Byte6T& rhs)
{
    return (lhs.data[0] == rhs.data[0]) && (lhs.data[1] == rhs.data[1])
             && (lhs.data[2] == rhs.data[2]) && (lhs.data[3] == rhs.data[3]);
}

GRACE_HOST_DEVICE bool operator==(const Byte7T& lhs, const Byte7T& rhs)
{
    return (lhs.data[0] == rhs.data[0]) && (lhs.data[1] == rhs.data[1])
             && (lhs.data[2] == rhs.data[2]) && (lhs.data[3] == rhs.data[3]);
}

GRACE_HOST_DEVICE bool operator==(const Byte8T& lhs, const Byte8T& rhs)
{
    return (lhs.data[0] == rhs.data[0]) && (lhs.data[1] == rhs.data[1])
             && (lhs.data[2] == rhs.data[2]) && (lhs.data[3] == rhs.data[3]);
}

GRACE_HOST_DEVICE bool operator==(const Byte16T& lhs, const Byte16T& rhs)
{
    return (lhs.data[0] == rhs.data[0]) && (lhs.data[1] == rhs.data[1])
             && (lhs.data[2] == rhs.data[2]) && (lhs.data[3] == rhs.data[3]);
}

GRACE_HOST_DEVICE bool operator!=(const Byte1T& lhs, const Byte1T& rhs)
{
    return !(lhs == rhs);
}

GRACE_HOST_DEVICE bool operator!=(const Byte2T& lhs, const Byte2T& rhs)
{
    return !(lhs == rhs);
}

GRACE_HOST_DEVICE bool operator!=(const Byte3T& lhs, const Byte3T& rhs)
{
    return !(lhs == rhs);
}

GRACE_HOST_DEVICE bool operator!=(const Byte4T& lhs, const Byte4T& rhs)
{
    return !(lhs == rhs);
}

GRACE_HOST_DEVICE bool operator!=(const Byte5T& lhs, const Byte5T& rhs)
{
    return !(lhs == rhs);
}

GRACE_HOST_DEVICE bool operator!=(const Byte6T& lhs, const Byte6T& rhs)
{
    return !(lhs == rhs);
}

GRACE_HOST_DEVICE bool operator!=(const Byte7T& lhs, const Byte7T& rhs)
{
    return !(lhs == rhs);
}

GRACE_HOST_DEVICE bool operator!=(const Byte8T& lhs, const Byte8T& rhs)
{
    return !(lhs == rhs);
}

GRACE_HOST_DEVICE bool operator!=(const Byte16T& lhs, const Byte16T& rhs)
{
    return !(lhs == rhs);
}
