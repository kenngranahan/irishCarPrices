"use strict";

const carData = [
                {county:'Carlow',make:'Toyota',model:'C-HR',vehicle_year:2019,mileage:55000,price:27495,transmission:'Automatic',engine:1.8,body:'Suv',fuel:'Hybrid',doors:4,mpg:62,owners:1,color:'Silver',tax:180,nct_month:'March',nct_year:2023},
                {county:'Dublin',make:'Audi',model:'A1',vehicle_year:2013,mileage:35000,price:14550,transmission:'Automatic',engine:1.4,body:'Hatchback',fuel:'Petrol',doors:5,mpg:NaN,owners:NaN,color:'Silver',tax:NaN,nct_month:'September',nct_year:2023},
                {county:'Louth',make:'Dacia',model:'Duster',vehicle_year:2019,mileage:127500,price:15500,transmission:'Manual',engine:1.5,body:'Other',fuel:'Diesel',doors:4,mpg:53,owners:1,color:'Grey',tax:200,nct_month:'None',nct_year:NaN},
                {county:'Dublin',make:'Bmw',model:'520',vehicle_year:2009,mileage:286462,price:5750,transmission:'Manual',engine:2,body:'Saloon',fuel:'Diesel',doors:4,mpg:46,owners:NaN,color:'Blue',tax:280,nct_month:'April',nct_year:2022},
                {county:'Cork',make:'Volkswagen',model:'Golf',vehicle_year:2013,mileage:169000,price:11999,transmission:'Manual',engine:1.6,body:'Hatchback',fuel:'Diesel',doors:5,mpg:62,owners:3,color:'Silver',tax:180,nct_month:'June',nct_year:2023}]

const scatterWidth = 1200
const scatterHeight = 800
const scatterContainer = d3.select('body').append('svg')
                            .attr('viewBox', [0,0,scatterWidth, scatterHeight])
const scatterGraph = scatterContainer.append('g')
                                      .attr('fill', 'none')
                                      .attr('stroke-linecap', 'round')
                                      .selectAll("circle")
                                      .data(carData)
                                      .join("circle")
                                      .attr("cx", d => d.mileage)
                                      .attr("cy", d => d.price)
                                      .attr("r", 1)
