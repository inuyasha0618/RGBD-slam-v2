add_library(myslam SHARED
        camera.cpp
        frame.cpp
        )

target_link_libraries(myslam ${THIRD_PARTY_LIBS})

ADD_EXECUTABLE(hello hello.cpp)