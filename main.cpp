#include "crow/scripts/crow_all.h"
//#include "MusicGenerator.cpp"
//using namespace std;

int main() {
	crow::SimpleApp app;

	CROW_ROUTE(app, "/<string>")([](std::string name) {
		auto page = crow::mustache::load("fancypage.html");
		crow::mustache::context ctx ({{"declaration", "Hello " + name + "!"}});
		return page.render(ctx);
	});

	CROW_ROUTE(app, "/<string>/<string>")([](std::string name1, std::string name2) {
		std::string name = name1 + " is cooler than " + name2;
		crow::mustache::context ctx ({{"declaration", name}});
		auto page = crow::mustache::load("fancypage.html");
		return page.render(ctx);
	});

	CROW_ROUTE(app, "/")([](){
		std::string declaration = "No user entered";
		crow::mustache::context ctx ({{"declaration", declaration}});
		auto page = crow::mustache::load("fancypage.html");
		return page.render(ctx);
	});

	app.port(18080).run();
}
