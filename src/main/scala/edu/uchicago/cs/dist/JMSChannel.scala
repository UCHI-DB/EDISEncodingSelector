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
 * under the License.
 *
 * Contributors:
 *     Hao Jiang - initial API and implementation
 */

package edu.uchicago.cs.dist

import java.util
import javax.jms._

import org.apache.activemq.ActiveMQConnectionFactory

/**
  *
  */

class JMSChannelRegistry(host: String) extends ChannelRegistry {

  protected val connectionFactory = new ActiveMQConnectionFactory(host)
  // Create a Connection
  protected val connection = connectionFactory.createConnection
  connection.start
  // Create a Session
  protected val session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE)


  override def find(name: String): Channel = {
    val dest = session.createQueue(name)
    new JMSChannel(name, session, dest)
  }
}

class JMSChannel(val id: String, val session: Session, val queue: Queue) extends Channel {

  val producer = session.createProducer(queue)
  val consumer = session.createConsumer(queue)

  def send(data: java.io.Serializable): Unit = {
    producer.send(session.createObjectMessage(data))
  }

  def listen(): java.io.Serializable = {
    consumer.receive().asInstanceOf[ObjectMessage].getObject
  }

  def listen(callback: (java.io.Serializable) => Unit): Unit = {
    consumer.setMessageListener(new MessageListener {
      def onMessage(message: Message): Unit = {
        callback(message.asInstanceOf[ObjectMessage].getObject)
      }
    })
  }
}
