import Content from "./Content-temp";

export default interface Section {
  id: number,
  name: string,
  value: string,
  image: string, 
  content: Content[]
}