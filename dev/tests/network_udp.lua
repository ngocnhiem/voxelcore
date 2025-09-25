math.randomseed(43172)
for i = 1, 15 do
    debug.print(string.format("iteration %s", i))
    local complete = false

    local server = network.udp_open(8645 + i, function (address, port, data, srv)
        debug.print(string.format("server received %s byte(s) from %s:%s", #data, address, port))
        srv:send(address, port, "pong")
    end)

    app.tick()
    network.udp_connect("localhost", 8645 + i, function (data)
        debug.print(string.format("client received %s byte(s) from server", #data))
        complete = true
    end, function (socket)
        debug.print("udp socket opened")
        start_coroutine(function()
            debug.print("udp data-sender started")
            for k = 1, 15 do
                local payload = ""
                for j = 1, 16 do
                    payload = payload .. math.random(0, 9)
                end
                socket:send(payload)
                debug.print(string.format("sent packet %s (%s bytes)", k, #payload))
                coroutine.yield()
            end
            socket:close()
        end, "udp-data-sender")
    end)

    app.sleep_until(function () return complete end, nil, 5)
    server:close()
end
