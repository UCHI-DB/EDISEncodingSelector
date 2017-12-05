/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License,
 *
 * Contributors:
 *     Hao Jiang - initial API and implementation
 *
 */

//package edu.uchicago.cs.ndnn.figure

//import javafx.application.Application
//import javafx.scene.chart.{LineChart, NumberAxis, XYChart}
//import javafx.scene.{Group, Scene}
//import javafx.stage.Stage

//import scala.collection.JavaConversions._
//import scala.collection.mutable.ArrayBuffer
//
/**
  * Created by harper on 5/10/17.
  */

//object Figure {
//  var current: Figure = null
//}
//
//class FigureApp extends Application {
//
//  override def start(primaryStage: Stage): Unit = {
//    if (Figure.current == null)
//      throw new IllegalArgumentException
//    Figure.current.config(primaryStage)
//    primaryStage.show()
//  }
//}
//
//abstract class Figure {
//
//  def config(stage: Stage): Unit
//
//  protected var width = 0
//  protected var height = 0
//
//  def show(width: Int = 800, height: Int = 600): Unit = {
//    this.width = width
//    this.height = height
//
//    Figure.current = this
//
//    Application.launch(classOf[FigureApp])
//  }
//}
//
//class LineFigure extends Figure {
//
//  protected var _title: String = ""
//
//  def title(title: String): LineFigure = {
//    this._title = title
//    this
//  }
//
//  protected var xlabel: String = ""
//
//  def xLabel(label: String): LineFigure = {
//    xlabel = label
//    this
//  }
//
//  protected var ylabel: String = ""
//
//  def yLabel(label: String): LineFigure = {
//    ylabel = label
//    this
//  }
//
//  protected val datas = new ArrayBuffer[(Seq[(Number, Number)], String)]
//
//  def addPlot(data: Seq[(Number, Number)], legend: String): LineFigure = {
//    datas += ((data, legend))
//    this
//  }
//
//
//  override def config(stage: Stage): Unit = {
//    val xaxis = new NumberAxis()
//    xaxis.setLabel(xlabel)
//    val yaxis = new NumberAxis()
//    yaxis.setLabel(ylabel)
//
//    val lineChart = new LineChart(xaxis, yaxis)
//    lineChart.setTitle(_title)
//    lineChart.setCreateSymbols(false)
//    lineChart.setPrefWidth(width)
//    lineChart.setPrefHeight(height)
//
//    datas.foreach(pair => {
//      val data = pair._1
//      val legend = pair._2
//      val plotData = new XYChart.Series[Number, Number]()
//      plotData.getData.addAll(data.map(point => new XYChart.Data(point._1, point._2)))
//      plotData.setName(legend)
//      lineChart.getData.add(plotData)
//    })
//
//    val root = new Group(lineChart)
//    //Creating a scene object
//    val scene: Scene = new Scene(root, width, height)
//    //Setting title to the Stage
//    stage.setTitle(_title)
//    //Adding scene to the stage
//    stage.setScene(scene)
//  }
//
//}
