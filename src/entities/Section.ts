import Content from "./content";

export default interface Section {
  id: number,
  name: string,
  value: string,
  image: string, 
  content: Content[]
}