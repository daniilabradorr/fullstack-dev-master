package com.example.bibliotecaApi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;

@SpringBootApplication
public class BibliotecaApiApplication extends ServletInitializer{

	public static void main(String[] args) {
		SpringApplication.run(BibliotecaApiApplication.class, args);
	}


	@Override
	protected SpringApplicationBuilder configure(SpringApplicationBuilder application){
		return application.sources(BibliotecaApiApplication.class);
	}
}
