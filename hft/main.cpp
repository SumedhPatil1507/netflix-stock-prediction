// Small, allocation-light L2 event loop. Input: CSV event_ns,symbol,bid,bid_size,ask,ask_size.
// It computes top-of-book OFI and can route capped IOC market orders to an already
// authenticated FIX acceptor. Session establishment and exchange connectivity belong
// to the venue-certified FIX engine; this process never logs on or sends live orders by default.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = std::intptr_t;
static void close_socket(socket_t s) { closesocket(static_cast<SOCKET>(s)); }
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static void close_socket(socket_t s) { close(s); }
#endif

struct Book { double bid=0, bid_size=0, ask=0, ask_size=0; };

static std::string env(const char* key, const char* fallback) {
    const char* value = std::getenv(key); return value && *value ? value : fallback;
}

static std::string fix_message(const std::string& symbol, char side, int qty, unsigned seq) {
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    const std::time_t seconds = static_cast<std::time_t>(ms / 1000);
    std::tm utc{};
#ifdef _WIN32
    if (const std::tm* value = std::gmtime(&seconds)) utc = *value;
#else
    gmtime_r(&seconds, &utc);
#endif
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d-%H:%M:%S", &utc);
    std::ostringstream body;
    body << "35=D\00149=" << env("FIX_SENDER_COMP_ID", "ALPHA")
         << "\00156=" << env("FIX_TARGET_COMP_ID", "BROKER") << "\00134=" << seq
         << "\00152=" << timestamp << "\00111=alpha-" << seq << "\00121=1\00155=" << symbol
         << "\00154=" << side << "\00160=" << timestamp << "\00138=" << qty
         << "\00140=1\00159=3\001"; // Market, IOC
    const std::string payload = body.str();
    std::ostringstream prefix;
    prefix << "8=FIX.4.4\0019=" << payload.size() << "\001" << payload;
    const std::string without_checksum = prefix.str();
    unsigned checksum = 0;
    for (unsigned char c : without_checksum) checksum = (checksum + c) % 256;
    char sum[8]; std::snprintf(sum, sizeof(sum), "%03u", checksum);
    return without_checksum + "10=" + sum + "\001";
}

static socket_t connect_fix(const char* host, const char* port) {
#ifdef _WIN32
    WSADATA data{}; if (WSAStartup(MAKEWORD(2,2), &data) != 0) return INVALID_SOCKET;
    hostent* resolved = gethostbyname(host);
    if (!resolved || !resolved->h_addr_list[0]) return -1;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<unsigned short>(std::atoi(port)));
    std::memcpy(&address.sin_addr, resolved->h_addr_list[0], sizeof(address.sin_addr));
    socket_t fd = static_cast<socket_t>(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (fd >= 0 && connect(static_cast<SOCKET>(fd), reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0)
        return fd;
    if (fd >= 0) close_socket(fd);
    return -1;
#else
    addrinfo hints{}; hints.ai_socktype = SOCK_STREAM; hints.ai_family = AF_UNSPEC;
    addrinfo* result = nullptr;
    if (getaddrinfo(host, port, &hints, &result) != 0) return -1;
    socket_t fd = -1;
    for (auto* address = result; address; address = address->ai_next) {
        fd = static_cast<socket_t>(socket(address->ai_family, address->ai_socktype, address->ai_protocol));
        if (fd >= 0 && connect(fd, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0) break;
        if (fd >= 0) close_socket(fd);
        fd = -1;
    }
    freeaddrinfo(result); return fd;
#endif
}

static bool send_fix(socket_t fd, const std::string& message) {
    std::size_t sent = 0;
    while (sent < message.size()) {
#ifdef _WIN32
        const int n = static_cast<int>(send(static_cast<SOCKET>(fd), message.data() + sent,
                                           static_cast<int>(message.size() - sent), 0));
#else
        const int n = static_cast<int>(send(fd, message.data() + sent,
                                           static_cast<int>(message.size() - sent), 0));
#endif
        if (n <= 0) return false;
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

int main() {
    const std::string threshold_value = env("OFI_ORDER_THRESHOLD", "1000");
    const std::string max_shares_value = env("MAX_ORDER_SHARES", "10");
    const std::string max_spread_value = env("MAX_SPREAD_BPS", "20");
    const double threshold = std::strtod(threshold_value.c_str(), nullptr);
    const int max_shares = std::max(0, std::atoi(max_shares_value.c_str()));
    const double max_spread_bps = std::strtod(max_spread_value.c_str(), nullptr);
    socket_t fix_fd = -1;
    const char* fix_host = std::getenv("FIX_HOST");
    const char* fix_port = std::getenv("FIX_PORT");
    if (fix_host && fix_port) {
        fix_fd = connect_fix(fix_host, fix_port);
        if (fix_fd < 0) std::cerr << "FIX connection unavailable; signals will be logged only\n";
    }
    std::unordered_map<std::string, Book> previous;
    std::string line;
    unsigned sequence = 1;
    while (std::getline(std::cin, line)) {
        const auto start = std::chrono::steady_clock::now();
        std::istringstream row(line);
        std::string ts, symbol, field;
        double bid, bid_size, ask, ask_size;
        if (!std::getline(row, ts, ',') || !std::getline(row, symbol, ',') ||
            !std::getline(row, field, ',')) continue;
        try { bid = std::stod(field); } catch (...) { continue; }
        if (!std::getline(row, field, ',')) continue;
        try { bid_size = std::stod(field); } catch (...) { continue; }
        if (!std::getline(row, field, ',')) continue;
        try { ask = std::stod(field); } catch (...) { continue; }
        if (!std::getline(row, field, ',')) continue;
        try { ask_size = std::stod(field); } catch (...) { continue; }
        if (symbol.empty() || bid <= 0 || ask < bid || bid_size < 0 || ask_size < 0) continue;
        const Book current{bid, bid_size, ask, ask_size};
        const auto found = previous.find(symbol);
        const Book prior = found == previous.end() ? current : found->second;
        const double bid_delta = bid > prior.bid ? bid_size : bid < prior.bid ? -prior.bid_size : bid_size - prior.bid_size;
        const double ask_delta = ask < prior.ask ? ask_size : ask > prior.ask ? -prior.ask_size : ask_size - prior.ask_size;
        const double ofi = bid_delta - ask_delta;
        const double mid = (bid + ask) / 2.0;
        const double spread_bps = mid > 0 ? (ask - bid) / mid * 10000.0 : 1e9;
        previous[symbol] = current;
        const auto elapsed = std::max<std::int64_t>(1, std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
        std::cout << ts << ',' << symbol << ",ofi=" << ofi << ",spread_bps=" << spread_bps
                  << ",processing_ns=" << elapsed << '\n';
        if (fix_fd >= 0 && std::abs(ofi) >= threshold && spread_bps <= max_spread_bps && max_shares > 0) {
            const char side = ofi > 0 ? '1' : '2';
            if (!send_fix(fix_fd, fix_message(symbol, side, max_shares, sequence++))) {
                std::cerr << "FIX send failed; disabling execution\n";
                close_socket(fix_fd); fix_fd = -1;
            }
        }
    }
    if (fix_fd >= 0) close_socket(fix_fd);
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
