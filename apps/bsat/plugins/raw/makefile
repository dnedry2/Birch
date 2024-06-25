CCPP=clang++

raw:
	$(CCPP) -O3 -std=c++17 -shared -fPIC -I ../../../../lib/imgui-docking/ -I ../../../../common/include obj/* *.cpp -o raw.so `pkg-config --cflags --libs freetype2`
	mkdir -p ../../../../bin/plugins
	mv *.so ../../../../bin/plugins

libs:
	mkdir -p obj
	$(CCPP) -std=c++11 -O3 -fPIC -c -I ../../../../lib/imgui-docking/ -I ../../../../lib/imgui-docking/misc/freetype ../../../../lib/imgui-docking/imgui.cpp ../../../../lib/imgui-docking/imgui_t*.cpp ../../../../lib/imgui-docking/imgui_w*.cpp ../../../../lib/imgui-docking/imgui_d*.cpp ../../../../lib/imgui-docking/misc/freetype/*.cpp `pkg-config --cflags --libs freetype2`
	mv *.o obj